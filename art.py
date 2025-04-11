"""
ART Module

This module defines the FuzzyARTMAP and FuzzyART models used for dynamic category discovery and
novelty detection. The ART (Adaptive Resonance Theory) models are designed for both supervised
and unsupervised learning, allowing the system to handle an expanding set of categories in a
lifelong learning context.

Classes:
- FuzzyARTMAP: A model for supervised learning and classification with category discovery.
- FuzzyART: A model for unsupervised clustering with category discovery.

Methods:
- __init__: Initializes the ART models with the specified parameters.
- forward: Performs the forward pass through the ART models.
- compute_match_scores: Computes match scores for input samples.
- update_category: Updates an existing category based on new input.
- create_new_category: Creates a new category for novel inputs with generic labels.
- adjust_vigilance: Adjusts the vigilance parameter for category match threshold.
- learn: Performs learning by updating or creating categories based on input and ground truth labels.
- get_current_categories: Returns the current number of categories.
- get_newly_discovered_categories: Retrieves generically labeled categories.
- assign_human_label_to_category: Assigns human-provided labels to generically labeled categories.
"""

import torch
import torch.nn as nn


class FuzzyARTMAP(nn.Module):
    """
    FuzzyARTMAP module for classification and novelty detection.
    """

    def __init__(
        self,
        input_dim,
        dynamic_categories,
        initial_vigilance=0.75,
        vigilance_increment=0.05,
        generic_label_prefix="curiosity",
        device=None,
    ):
        """
        Initializes the FuzzyARTMAP module.

        Parameters:
        - input_dim (int): Dimension of the input features.
        - dynamic_categories (int): Maximum number of categories.
        - initial_vigilance (float): Initial vigilance parameter.
        - vigilance_increment (float): Increment step for the vigilance parameter.
        - generic_label_prefix (str): Prefix for generically labeled new categories.
        """
        super(FuzzyARTMAP, self).__init__()
        self.input_dim = input_dim
        self.dynamic_categories = dynamic_categories
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.categories = torch.randn(dynamic_categories, input_dim, device=self.device)
        self.vigilance = initial_vigilance
        self.vigilance_increment = vigilance_increment
        self.generic_label_prefix = generic_label_prefix
        self.generic_category_counter = 0
        self.labels = [None] * dynamic_categories
        self.to(self.device)

    def forward(self, x):
        """
        Forward pass through the FuzzyARTMAP module.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Match scores for each category.
        - torch.Tensor: Indices of the most relevant categories.
        """
        x = x.to(self.device)
        match_scores = torch.zeros(
            x.size(0), self.dynamic_categories, device=self.device
        )
        for i, sample in enumerate(x):
            match_scores[i] = self.compute_match_scores(sample)
        max_scores, indices = torch.max(match_scores, dim=1)
        return match_scores, indices

    def compute_match_scores(self, sample):
        """
        Computes match scores for a single sample.

        Parameters:
        - sample (torch.Tensor): Single input sample.

        Returns:
        - torch.Tensor: Match scores for the sample.
        """
        match_scores = torch.zeros(self.dynamic_categories, device=self.device)

        for i, category in enumerate(self.categories):
            category = category.to(self.device)

            match = torch.sum(torch.min(sample, category)) / torch.sum(sample)
            if match >= self.vigilance:
                match_scores[i] = match

        return match_scores

    def update_category(self, sample, index, label=None):
        """
        Updates an existing category with a new sample.

        Parameters:
        - sample (torch.Tensor): New input sample.
        - index (int): Index of the category to update.
        - label (str): Optional label to assign to the category.
        """
        sample = sample.to(self.device)
        self.categories[index] = self.categories[index].to(self.device)

        self.categories[index] = torch.min(
            self.categories[index].detach().clone(), sample.detach().clone()
        )

        if label:
            self.labels[index] = label

    def create_new_category(self, sample):
        """
        Creates a new category for a novel input sample with a generic label.

        Parameters:
        - sample (torch.Tensor): Novel input sample.
        """
        sample = sample.to(self.device)
        if len(self.categories) < self.dynamic_categories:
            new_category_index = len(self.categories)
            self.categories = torch.cat((self.categories, sample.unsqueeze(0)), dim=0)
            self.labels.append(
                f"{self.generic_label_prefix}-{self.generic_category_counter}"
            )
        else:
            new_category_index = torch.argmin(torch.sum(self.categories, dim=1))
            self.labels[new_category_index] = (
                f"{self.generic_label_prefix}-{self.generic_category_counter}"
            )
        self.categories[new_category_index] = sample
        self.generic_category_counter += 1

    def adjust_vigilance(self, increment=True):
        """
        Adjusts the vigilance parameter.

        Parameters:
        - increment (bool): If True, increments the vigilance parameter, otherwise decrements it.
        """
        if increment:
            self.vigilance += self.vigilance_increment
        else:
            self.vigilance = max(0.0, self.vigilance - self.vigilance_increment)

    def learn(self, x, labels):
        """
        Learns from the input samples and updates categories or creates new ones as necessary.

        Parameters:
        - x (torch.Tensor): Input tensor.
        - labels (torch.Tensor): Ground truth labels for the input tensor.
        """
        x = x.to(self.device)
        match_scores, indices = self.forward(x)
        for i, (sample, label) in enumerate(zip(x, labels)):
            if labels is not None:
                label = labels[i]
            if match_scores[i, indices[i]] >= self.vigilance:
                # print(
                #     f"Updating category at index {indices[i]} with sample from label {label if labels is not None else 'N/A'}"
                # )
                self.update_category(sample, indices[i], label)
            else:
                # print(
                #     f"Creating new category for sample from label {label if labels is not None else 'N/A'} with initial vigilance {self.vigilance}"
                # )
                self.create_new_category(sample)
                self.adjust_vigilance(increment=True)
                # print(f"New vigilance after increment: {self.vigilance}")

    def get_current_categories(self):
        """
        Returns the current number of categories.
        """
        return len(self.categories)

    def get_newly_discovered_categories(self):
        """
        Retrieves categories that were discovered during model deployment and were assigned generic labels.

        Returns:
        - list: A list of generically labeled categories.
        """
        return [
            (i, label)
            for i, label in enumerate(self.labels)
            if label.startswith(self.generic_label_prefix)
        ]

    def assign_human_label_to_category(self, index, human_label):
        """
        Assigns a human-provided label to a generically labeled category.

        Parameters:
        - index (int): Index of the category to be renamed.
        - human_label (str): The new label provided by a human expert.
        """
        if self.labels[index].startswith(self.generic_label_prefix):
            self.labels[index] = human_label


class FuzzyART(nn.Module):
    """
    FuzzyART module for unsupervised clustering.
    """

    def __init__(
        self,
        input_dim,
        dynamic_categories,
        initial_vigilance=0.75,
        vigilance_increment=0.05,
        generic_label_prefix="curiosity",
        device=None,
    ):
        """
        Initializes the FuzzyART module.

        Parameters:
        - input_dim (int): Dimension of the input features.
        - dynamic_categories (int): Maximum number of categories.
        - initial_vigilance (float): Initial vigilance parameter.
        - vigilance_increment (float): Increment step for the vigilance parameter.
        - generic_label_prefix (str): Prefix for generically labeled new categories.
        """
        super(FuzzyART, self).__init__()
        self.input_dim = input_dim
        self.dynamic_categories = dynamic_categories
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.categories = torch.randn(
            dynamic_categories, input_dim
        )  # Initialize categories randomly
        self.vigilance = initial_vigilance  # Initial vigilance parameter
        self.vigilance_increment = vigilance_increment  # Vigilance increment step
        self.generic_label_prefix = generic_label_prefix
        self.generic_category_counter = 0
        self.labels = [""] * dynamic_categories  # Placeholder for category labels
        self.to(self.device)

    def forward(self, x):
        """
        Forward pass through the FuzzyART module.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Match scores for each category.
        - torch.Tensor: Indices of the most relevant categories.
        """
        x = x.to(self.device)
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension (batch_size, {self.input_dim}), but got {x.shape}"
            )
        match_scores = torch.zeros(x.size(0), len(self.categories), device=self.device)
        for i, sample in enumerate(x):
            match_scores[i] = self.compute_match_scores(sample)
        max_scores, indices = torch.max(match_scores, dim=1)
        return match_scores, indices

    def compute_match_scores(self, sample):
        """
        Computes match scores for a single sample.

        Parameters:
        - sample (torch.Tensor): Single input sample.

        Returns:
        - torch.Tensor: Match scores for the sample.
        """
        match_scores = torch.zeros(self.dynamic_categories)
        for i, category in enumerate(self.categories):
            match = torch.sum(torch.min(sample, category)) / torch.sum(sample)
            if match >= self.vigilance:
                match_scores[i] = match
        return match_scores

    def update_category(self, sample, index):
        """
        Updates an existing category with a new sample.

        Parameters:
        - sample (torch.Tensor): New input sample.
        - index (int): Index of the category to update.
        """
        sample = sample.to(self.device)
        self.categories = self.categories.to(self.device)
        self.categories[index] = torch.min(self.categories[index], sample)

    def create_new_category(self, sample):
        """
        Creates a new category for a novel input sample with a generic label.

        Parameters:
        - sample (torch.Tensor): Novel input sample.
        """
        sample = sample.to(self.device)
        self.categories = self.categories.to(self.device)
        if len(self.categories) < self.dynamic_categories:
            new_category_index = len(self.categories)
            self.categories = torch.cat((self.categories, sample.unsqueeze(0)), dim=0)
            self.labels.append(
                f"{self.generic_label_prefix}-{self.generic_category_counter}"
            )
        else:
            new_category_index = torch.argmin(torch.sum(self.categories, dim=1))
            self.labels[new_category_index] = (
                f"{self.generic_label_prefix}-{self.generic_category_counter}"
            )
        self.categories[new_category_index] = sample
        self.generic_category_counter += 1

    def adjust_vigilance(self, increment=True):
        """
        Adjusts the vigilance parameter.

        Parameters:
        - increment (bool): If True, increments the vigilance parameter, otherwise decrements it.
        """
        if increment:
            self.vigilance += self.vigilance_increment
        else:
            self.vigilance = max(0.0, self.vigilance - self.vigilance_increment)

    def learn(self, x):
        """
        Learns from the input samples and updates categories or creates new ones as necessary.

        Parameters:
        - x (torch.Tensor): Input tensor.
        """
        x = x.to(self.device)
        self.categories = self.categories.to(self.device)
        match_scores, indices = self.forward(x)
        for i, sample in enumerate(x):
            if match_scores[i, indices[i]] >= self.vigilance:
                self.update_category(sample, indices[i])
            else:
                self.create_new_category(sample)
                self.adjust_vigilance(increment=True)

    def get_current_categories(self):
        """
        Returns the current number of categories.
        """
        self.categories = self.categories.to(
            self.device
        )  # Ensure categories are on the same device
        return len(self.categories)

    def get_newly_discovered_categories(self):
        """
        Retrieves categories that were discovered during model deployment and were assigned generic labels.

        Returns:
        - list: A list of generically labeled categories.
        """
        return [
            (i, label)
            for i, label in enumerate(self.labels)
            if label.startswith(self.generic_label_prefix)
        ]

    def assign_human_label_to_category(self, index, human_label):
        """
        Assigns a human-provided label to a generically labeled category.

        Parameters:
        - index (int): Index of the category to be renamed.
        - human_label (str): The new label provided by a human expert.
        """
        if self.labels[index].startswith(self.generic_label_prefix):
            self.labels[index] = human_label
