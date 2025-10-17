import os
import numpy as np
import torch
from collections import defaultdict
from transformers import TrainerCallback
from utils.analyzer3 import ViTAttentionAnalyzer  # Ensure this module is available
import time
import random

def get_samples_by_class(prepared_ds, testsize):
    start_time = time.time()
    class_samples = defaultdict(list)

    for sample in prepared_ds["train"]:
        label = sample['labels']
        if isinstance(label, torch.Tensor):
            label = label.item()
        class_samples[label].append(sample['pixel_values'])
    print(len(class_samples))
    selected_samples = []
    for label in class_samples:
        if len(class_samples[label]) >= testsize:
            selected_samples.extend(random.sample(class_samples[label], testsize))
        else:
            selected_samples.extend(class_samples[label])

    pixel_value_samples = torch.stack(selected_samples)
    total_sample_size = len(selected_samples)

    print(total_sample_size)
    return pixel_value_samples, total_sample_size



class DistributionUpdater:
    @staticmethod
    def update_distribution(cumulative_distribution, new_array):
        start_time = time.time()
        for sample in range(new_array.shape[0]):
            for layer in range(new_array.shape[1]):
                for head in range(new_array.shape[2]):
                    values, counts = np.unique(new_array[sample, layer, head], return_counts=True)
                    for value, count in zip(values, counts):
                        cumulative_distribution[sample, layer, head, value - 1] += count
        print(f"Time taken for update_distribution: {time.time() - start_time:.4f} seconds")
        return cumulative_distribution


class AttentionLoggerCallback:
    def __init__(self, model, prepared_ds, log_dir="./logs", max_steps=50, save_interval=1):
        self.model = model
        self.prepared_ds = prepared_ds
        self.log_dir = log_dir
        self.max_steps = max_steps
        self.testsizeini = 3

        start_time = time.time()
        self.pixel_value_samples, self.testsize = get_samples_by_class(self.prepared_ds, self.testsizeini)
        print(f"Initialization: testsize = {self.testsize}")
        print(f"Time taken for initial sample extraction: {time.time() - start_time:.4f} seconds")

        self.save_interval = save_interval
        self.current_step = 0

        #self.median = np.zeros((max_steps, 12, 12), dtype=np.float32)
        self.median = np.zeros((max_steps, 12, 12), dtype=np.float32)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.analyzer = ViTAttentionAnalyzer(self.model)

    def on_step_end(self):
        if self.current_step < self.max_steps:
            self.median[self.current_step] = self.analyzer.get_attention_weights_vit(self.pixel_value_samples)

        if self.current_step == self.max_steps-1:
            print("Reached max steps, saving totals.")

            self.save_totals()

        self.current_step += 1

    def save_totals(self):
        total_median = os.path.join(self.log_dir, 'total_median_dinov2.npy')
        np.save(total_median, self.median)
        print("stored")

