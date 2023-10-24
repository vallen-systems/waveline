"""
Skeleton pipeline with data acquisition, feature extraction, classification and OPC UA output.

The pipeline can run on an external PC or directly on the linWave (using the IP 127.0.0.1).
Please contact Vallen System <software@vallen.de> for more details.

Make sure to install `asyncua` for the OPC UA server:

    pip install asyncua>=0.9
"""

import argparse
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from asyncua import Server
from waveline import LinWave

SAMPLERATE = 1_000_000  # 1 MHz
BLOCKSIZE = 524288  # = 2^19 (use power of 2 for efficient FFT computation)
RANGE_INDEX = 0  # 50 mV

logging.basicConfig(level=logging.INFO)


@dataclass
class Features:
    """Feature vector."""

    rms: float
    crest: float


@dataclass
class Result:
    """Classification result."""

    ok: bool


@dataclass
class Output:
    """Container for OPC UA outputs."""

    features: Features
    result: Result


def extract_features(signal: np.ndarray) -> Features:
    """Feature extraction."""
    peak = np.max(np.abs(signal))
    rms = np.sqrt(np.mean(signal**2))
    return Features(
        rms=rms,
        crest=peak / rms,
    )


def classify(features: Features) -> Result:
    """Inference based on feature vector, just a placeholder for your logic/model."""
    if features.rms >= 1e-3:  # >= 1 mV
        return Result(ok=False)
    return Result(ok=True)


async def pipeline(ip: str, queue: asyncio.Queue):
    """Pipeline: data acquisition, feature extraction and classification."""
    async with LinWave(ip) as lw:
        print(await lw.get_info())
        await lw.set_range_index(0, range_index=RANGE_INDEX)
        await lw.set_tr_decimation(0, factor=int(lw.MAX_SAMPLERATE / SAMPLERATE))
        await lw.set_filter(0, highpass=100e3, lowpass=500e3, order=8)

        stream = lw.stream(channel=1, blocksize=BLOCKSIZE)  # open streaming port before start acq
        await lw.start_acquisition()

        with ThreadPoolExecutor(max_workers=1) as pool:
            loop = asyncio.get_event_loop()
            # read streaming data in blocks
            async for time, y in stream:
                # execute (longer) blocking operations in the thread pool -> don't block event loop
                features = await loop.run_in_executor(pool, extract_features, y)
                result = await loop.run_in_executor(pool, classify, features)
                print(time, features, result)
                # push output to queue
                queue.put_nowait(Output(features=features, result=result))

        await lw.stop_acquisition()


async def opcua_server(queue: asyncio.Queue):
    """OPC UA server."""
    # setup server
    server = Server()
    await server.init()
    server.set_endpoint("opc.tcp://0.0.0.0:4840")
    server.set_server_name("linWave OPC UA server")

    # setup nodes
    uri = "https://www.vallen.de"
    idx = await server.register_namespace(uri)
    root_node = server.nodes.objects
    features_node = await root_node.add_folder(idx, "features")
    rms_node = await features_node.add_variable(idx, "features:rms", 0.0)
    crest_node = await features_node.add_variable(idx, "features:crest", 0.0)
    result_node = await root_node.add_folder(idx, "result")
    ok_node = await result_node.add_variable(idx, "result:ok", True)

    # wait for data and write to nodes
    async with server:
        while True:
            # get output from queue
            output: Output = await queue.get()
            queue.task_done()
            await rms_node.write_value(float(output.features.rms))
            await crest_node.write_value(float(output.features.crest))
            await ok_node.write_value(bool(output.result.ok))


async def main(ip: str):
    """Launch pipeline and OPC UA server."""
    output_queue: asyncio.Queue = asyncio.Queue()  # decouple acquisition/processing and io

    await asyncio.gather(
        pipeline(ip, output_queue),
        opcua_server(output_queue),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="linwave_pipeline")
    parser.add_argument("ip", help="IP address of linWave device")
    args = parser.parse_args()

    asyncio.run(main(args.ip))
