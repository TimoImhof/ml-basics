from multiprocessing import Process
import time
import os
from worker import BPEWorker
from bpe_logic import BytePairEncoderConfig
from dataclasses import dataclass


@dataclass
class BPEManagerConfig:
    num_workers: int = 4
    auto_determine: bool = False


class BPEManager:
    def __init__(
        self, bpe_config: BytePairEncoderConfig, manager_config: BPEManagerConfig
    ):
        self.bpe_config = bpe_config
        self.manager_config = manager_config
        self.workers = []

    def _determine_num_workers_based_on_system_stats(self) -> int:
        return os.cpu_count() or 4

    def _create_workforce(self):
        if self.manager_config.auto_determine:
            num_workers = self._determine_num_workers_based_on_system_stats()
        for _ in range(num_workers):
            self.workers.append(BPEWorker(self.bpe_config))

    def _split_corpus_in_chunks(self, corpus: str):
        pass

    def _pack_chunks_in_queue(self):
        pass

    def _work_off_queue(self):
        pass

    def launch(self):
        self._create_workforce()


if __name__ == "__main__":
    pass
