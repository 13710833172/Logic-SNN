import unittest

from train_ddp import StridedDistributedSampler


class DistributedSamplerTests(unittest.TestCase):
    def test_evaluation_shards_cover_dataset_once(self):
        dataset = list(range(10))
        shards = [
            list(StridedDistributedSampler(dataset, rank, world_size=3))
            for rank in range(3)
        ]

        flattened = [index for shard in shards for index in shard]
        self.assertEqual(sorted(flattened), list(range(len(dataset))))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual([len(shard) for shard in shards], [4, 3, 3])


if __name__ == "__main__":
    unittest.main()
