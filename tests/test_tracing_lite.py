# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import sys
from typing import Iterator

from monkeytype.tracing import (
    CallTrace,
    CallTraceLogger,
    trace_calls,
)
from monkeytype.typing import NoneType

print(f"sys.version: {sys.version}")

class TraceCollector(CallTraceLogger):
    def __init__(self):
        super(TraceCollector, self).__init__()
        self.traces = []
        self.flushed = False

    def log(self, trace: CallTrace):
        self.traces.append(trace)

    def flush(self):
        self.flushed = True


def squares(n: int) -> Iterator[int]:
    for i in range(n):
        yield i * i


def test_generator_trace():
    collector = TraceCollector()
    with trace_calls(collector, max_typed_dict_size=0):
        for _ in squares(3):
            pass
    print(collector.traces)
    if collector.traces != [CallTrace(squares, {'n': int}, NoneType, int)]:
        raise ValueError("mismatch")


if __name__ == "__main__":
    test_generator_trace()
