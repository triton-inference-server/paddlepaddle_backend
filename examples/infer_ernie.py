#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
import json
import argparse
import numpy as np

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


FLAGS = None


def parse_model_http(model_metadata, model_config):
    return model_metadata['inputs'], model_metadata['outputs']


def postprocess(results, output_metadata, batch_size):
    """
    Post-process results to show classifications.
    """

    output_array = results.as_numpy(output_metadata[0]['name'])
    return np.argmax(output_array, axis=1)


def read_input(filename):
    with open(filename) as file:
        data = json.load(file)
        return data


def requestGenerator(input_metadata, output_metadata, FLAGS, input_data):

    # Set the input data
    inputs = list()

    for input_ in input_metadata:
        input_name = input_['name']
        runtime_data = input_data[input_name]
        data = np.asarray(runtime_data['content'], dtype=np.int32)
        data = data.reshape(runtime_data['shape'])
        inputs.append(
          httpclient.InferInput(input_name, data.shape, input_['datatype']))
        inputs[-1].set_data_from_numpy(data, binary_data=True)

    outputs = list()
    for output in output_metadata:
        outputs.append(
            httpclient.InferRequestedOutput(output['name'],
                                          binary_data=True))

    yield inputs, outputs, FLAGS.model_name, FLAGS.model_version


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                      '--verbose',
                      action="store_true",
                      required=False,
                      default=False,
                      help='Enable verbose output')
    parser.add_argument('-m',
                        '--model-name',
                        type=str,
                        required=True,
                        help='Name of model')
    parser.add_argument(
        '-x',
        '--model-version',
        type=str,
        required=False,
        default="",
        help='Version of model. Default is to use latest version.')
    parser.add_argument('-b',
                        '--batch-size',
                        type=int,
                        required=False,
                        default=1,
                        help='Batch size. Default is 1.')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument('-i',
                        '--protocol',
                        type=str,
                        required=False,
                        choices=['HTTP'],
                        default='HTTP',
                        help='Protocol used to communicate with ' +
                        'the inference service. Default is HTTP.')
    parser.add_argument('image_filename',
                        type=str,
                        nargs='?',
                        default=None,
                        help='Input image / Input folder.')
    FLAGS = parser.parse_args()

    try:
        triton_client = httpclient.InferenceServerClient(
            url=FLAGS.url, verbose=FLAGS.verbose, concurrency=1)
    except Exception as exception:
        print("client creation failed: " + str(exception))
        sys.exit(1)

    # Make sure the model matches our requirements, and get some
    # properties of the model that we need for preprocessing
    try:
        model_metadata = triton_client.get_model_metadata(
            model_name=FLAGS.model_name, model_version=FLAGS.model_version)
    except InferenceServerException as e:
        print("failed to retrieve the metadata: " + str(e))
        sys.exit(1)

    try:
        model_config = triton_client.get_model_config(
            model_name=FLAGS.model_name, model_version=FLAGS.model_version)
    except InferenceServerException as e:
        print("failed to retrieve the config: " + str(e))
        sys.exit(1)

    requests = []
    responses = []
    request_ids = []

    input_metadata, output_metadata = parse_model_http(model_metadata, model_config)

    json_data = read_input(f'data/perf.{FLAGS.batch_size}.json')
    input_data = json_data['data']
    if 'ground_truth' in json_data:
        ground_truth = json_data['ground_truth']

    for idx, batch_data in enumerate(input_data):
        try:
            for inputs, outputs, model_name, model_version in requestGenerator(input_metadata, output_metadata, FLAGS, batch_data):
                responses.append(
                    triton_client.infer(FLAGS.model_name,
                        inputs,
                        request_id=str(idx),
                        model_version=FLAGS.model_version,
                        outputs=outputs))
        except InferenceServerException as e:
            print("inference failed: " + str(e))
            sys.exit(1)

    results = list()
    for response in responses:
        this_id = response.get_response()["id"]
        results.extend(postprocess(response, output_metadata, FLAGS.batch_size))

    if 'ground_truth' in json_data:
        print('Accuracy:', sum(np.asarray(ground_truth) == np.asarray(results))/len(ground_truth))
