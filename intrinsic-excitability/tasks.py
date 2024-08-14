from abc import ABC, abstractmethod
import torch


class Task(ABC):
    @abstractmethod
    def generate_data(self):
        """Every task must implement a generate_data method"""

    @abstractmethod
    def input_dimensionality(self):
        """Return the input dimensionality of the task"""

    @abstractmethod
    def output_dimensionality(self):
        """Return the output dimensionality of the task"""


class GoNogo(Task):
    def __init__(self, D, sigma, stim_time=50, delay_time=10, decision_time=10):
        self.D = D
        self.stim_time = stim_time
        self.delay_time = delay_time
        self.decision_time = decision_time
        self.sigma = sigma

        # Generate a random "cursor" vector
        self.cursor = self._generate_cursor()

    def _generate_cursor(self):
        """Generate a random unit norm cursor vector"""
        cursor = torch.randn(self.D)
        return cursor / torch.norm(cursor)

    def _generate_source(self, B, source_strength, source_floor):
        """Generate a random source strength for each batch element with a floor"""
        s = source_strength * torch.randn(B)
        sign_s = torch.sign(s)
        return torch.clamp(torch.abs(s), min=source_floor) * sign_s

    def input_dimensionality(self):
        """return dimensionality of task input"""
        return self.D + 1

    def output_dimensionality(self):
        return 2

    def sequence_length(self):
        return self.stim_time + self.delay_time + self.decision_time

    def generate_data(self, B, sigma=None, delay_time=None, source_strength=1.0, source_floor=0.0):
        """Generate a batch of data with B batch elements"""
        # Generate random source strength
        source = self._generate_source(B, source_strength, source_floor)

        # Determine noise strength
        sigma = sigma or self.sigma
        delay_time = delay_time or self.delay_time

        # Generate noise
        noise = torch.randn(B, self.stim_time, self.D) * sigma

        # Add signal to noise
        X = noise + source.unsqueeze(1).unsqueeze(2) * self.cursor.unsqueeze(0).unsqueeze(1)
        X = torch.cat([X, torch.zeros(B, delay_time + self.decision_time, self.D)], dim=1)

        # Measure the empirical source strength with noise, and generate labels
        s_empirical = torch.mean(X @ self.cursor, dim=1)
        labels = (s_empirical > 0).long()  # (targets - 1 means Go, 0 means NoGo)

        # Add fixation trigger
        fixation = torch.zeros((B, self.stim_time + delay_time + self.decision_time, 1))
        fixation[:, : self.stim_time + delay_time] = 1.0
        X = torch.cat([X, fixation], dim=2)

        # Build target
        predecision = torch.zeros((B, self.stim_time + delay_time, self.output_dimensionality()))
        postdecision = torch.zeros(B, self.output_dimensionality())
        postdecision = torch.scatter(postdecision, 1, labels.unsqueeze(1), 1).unsqueeze(1).expand(-1, self.decision_time, -1)
        target = torch.cat([predecision, postdecision], dim=1)

        # Data params
        params = dict(
            sigma=sigma,
            s_empirical=s_empirical,
        )
        return X, target, params


class ContextualGoNogo(Task):
    def __init__(self, D, sigma, num_contexts=2, stim_time=40, delay_time=10, decision_time=10):
        self.D = D
        self.num_contexts = num_contexts
        self.stim_time = stim_time
        self.delay_time = delay_time
        self.decision_time = decision_time
        self.sigma = sigma

        assert self.num_contexts > 1, "Number of contexts must be greater than 1"
        assert self.num_contexts < self.D, "Number of contexts must be less than the dimensionality"

        # Generate a random set of orthogonal "cursor" vectors
        self.cursors = torch.linalg.qr(torch.randn(self.D, self.D)).Q[:, : self.num_contexts].T

    def _generate_source(self, B, source_strength, source_floor):
        """Generate a random source strength for each batch element with a floor"""
        s = source_strength * torch.randn(B)
        sign_s = torch.sign(s)
        return torch.clamp(torch.abs(s), min=source_floor) * sign_s

    def input_dimensionality(self):
        """return dimensionality of task input"""
        return self.D + 1 + self.num_contexts

    def output_dimensionality(self):
        return 2

    def sequence_length(self):
        return self.stim_time + self.delay_time + self.decision_time

    def generate_data(self, B, sigma=None, delay_time=None, source_strength=1.0, source_floor=0.0):
        """Generate a batch of data with B batch elements"""
        # Generate random source strength
        sources = torch.stack([self._generate_source(B, source_strength, source_floor) for _ in range(self.num_contexts)], dim=1)
        cursor_signal = torch.sum(sources.unsqueeze(2) * self.cursors.unsqueeze(0), dim=1)

        # Determine noise strength
        sigma = sigma or self.sigma
        delay_time = delay_time or self.delay_time

        # Generate noise
        noise = torch.randn(B, self.stim_time, self.D) * sigma

        # Add signal to noise
        X = noise + cursor_signal.unsqueeze(1)
        X = torch.cat([X, torch.zeros(B, delay_time + self.decision_time, self.D)], dim=1)

        # Choose a random context for each batch element
        context_idx = torch.randint(self.num_contexts, (B,))

        # Measure the empirical source strength with noise, and generate labels
        s_empirical = torch.mean(X @ self.cursors.T, dim=1)
        s_target = torch.gather(s_empirical, 1, context_idx.unsqueeze(1)).squeeze(1)

        labels = (s_target > 0).long()  # (targets - 1 means Go, 0 means NoGo)

        # Add fixation trigger
        fixation = torch.zeros((B, self.stim_time + delay_time + self.decision_time, 1))
        fixation[:, : self.stim_time + delay_time] = 1.0
        X = torch.cat([X, fixation], dim=2)

        # Add context index to X
        context_inputs = torch.scatter(torch.zeros(B, self.num_contexts), 1, context_idx.unsqueeze(1), 1)

        X = torch.cat([X, context_inputs.unsqueeze(1).expand(-1, X.size(1), -1)], dim=2)

        # Build target
        predecision = torch.zeros((B, self.stim_time + delay_time, self.output_dimensionality()))
        postdecision = torch.zeros(B, self.output_dimensionality())
        postdecision = torch.scatter(postdecision, 1, labels.unsqueeze(1), 1).unsqueeze(1).expand(-1, self.decision_time, -1)
        target = torch.cat([predecision, postdecision], dim=1)

        # Data params
        params = dict(
            s_empirical=s_empirical,
            context_idx=context_idx,
            labels=labels,
        )
        return X, target, params


# class VarGoNogo(GoNogo):
#     """Go-Nogo with a variable threshold"""

#     def __init__(self, D, T, sigma):
#         self.D = D
#         self.T = T
#         self.sigma = sigma

#         # Generate a random "cursor" vector
#         self.cursor = self._generate_cursor()

#     def input_dimensionality(self):
#         """return dimensionality of task input"""
#         return self.D + 1

#     def output_dimensionality(self):
#         return 2

#     def generate_data(self, B, sigma=None, source_strength=1.0, source_floor=0.0, data_norm=0.1):
#         """Generate a batch of data with B batch elements"""
#         # Generate random source strength
#         source = self._generate_source(B, source_strength, source_floor)

#         # Generate random threshold and shift sources (so floor is consistent with threshold)
#         threshold = torch.randn(B)
#         source = source + threshold

#         # Determine noise strength
#         sigma = sigma or self.sigma

#         # Generate noise
#         noise = torch.randn(B, self.T, self.D) * sigma

#         # Generate normalized data
#         X = noise + source.unsqueeze(1).unsqueeze(2) * self.cursor.unsqueeze(0).unsqueeze(1)
#         X = X / torch.norm(X, dim=2, keepdim=True) * data_norm

#         # Measure the empirical source strength with noise, and generate labels
#         s_empirical = torch.mean(X @ self.cursor, dim=1)
#         labels = (s_empirical > threshold).long()  # (targets - 1 means Go, 0 means NoGo)

#         # Add threshold to X
#         X = torch.cat([X, threshold.unsqueeze(1).unsqueeze(2).expand(-1, self.T, 1)], dim=2)

#         # Data params
#         params = dict(
#             s_empirical=s_empirical,
#             threshold=threshold,
#         )
#         return X, labels, params


# class ContextualGoNogo(GoNogo):
#     def __init__(self, D, sigma, num_contexts=2):
#         self.D = D
#         self.sigma = sigma
#         self.num_contexts = num_contexts

#         assert self.num_contexts > 1, "Number of contexts must be greater than 1"
#         assert self.num_contexts < self.D, "Number of contexts must be less than the dimensionality"

#         # Generate a random "cursor" vector
#         self.cursors = torch.stack([self._generate_cursor() for _ in range(self.num_contexts)])

#         # Ensure that the cursor vectors are linearly independent
#         while not torch.linalg.matrix_rank(self.cursors) == self.num_contexts:
#             self.cursors = torch.stack([self._generate_cursor() for _ in range(self.num_contexts)])

#     def input_dimensionality(self):
#         """return dimensionality of task input"""
#         return self.D + self.num_contexts

#     def output_dimensionality(self):
#         return 2

#     def generate_data(self, B, T, sigma=None, source_strength=1.0, source_floor=0.0, data_norm=0.1):
#         """Generate a batch of data with B batch elements"""
#         # Generate random source strength
#         sources = torch.stack([self._generate_source(B, source_strength, source_floor) for _ in range(self.num_contexts)], dim=1)
#         cursor_signal = torch.sum(sources.unsqueeze(2) * self.cursors.unsqueeze(0), dim=1)

#         # Determine noise strength
#         sigma = sigma or self.sigma

#         # Generate noise
#         noise = torch.randn(B, T, self.D) * sigma

#         # Generate normalized data
#         X = noise + cursor_signal.unsqueeze(1)
#         X = X / torch.norm(X, dim=2, keepdim=True) * data_norm

#         # Choose a random context for each batch element
#         context_idx = torch.randint(self.num_contexts, (B,))

#         # Measure the empirical source strength with noise, and generate labels
#         s_empirical = torch.mean(X @ self.cursors.T, dim=1)
#         s_target = torch.gather(s_empirical, 1, context_idx.unsqueeze(1)).squeeze(1)

#         labels = (s_target > 0).long()  # (targets - 1 means Go, 0 means NoGo)

#         # Add context index to X
#         context_inputs = torch.scatter(torch.zeros(B, self.num_contexts), 1, context_idx.unsqueeze(1), 1)

#         X = torch.cat([X, context_inputs.unsqueeze(1).expand(-1, T, -1)], dim=2)

#         # Data params
#         params = dict(
#             s_empirical=s_empirical,
#             context_idx=context_idx,
#         )
#         return X, labels, params


# class DelayMatchSample(Task):
#     def __init__(self, D, num_stimuli=2, stim_time=10, delay_time=10, decision_time=10, as_spikes=True, strength=0.5):
#         self.D = D
#         self.num_stimuli = num_stimuli
#         self.stim_time = stim_time
#         self.delay_time = delay_time
#         self.decision_time = decision_time
#         self.as_spikes = as_spikes
#         self.strength = strength

#     def input_dimensionality(self):
#         """return dimensionality of task input"""
#         return self.num_stimuli * self.D

#     def decision_start_time(self):
#         """return time at which decision starts"""
#         return 2 * self.stim_time + self.delay_time

#     def generate_data(self, B):
#         """Generate a batch of data with B batch elements"""
#         # Generate random stimuli
#         stimuli = torch.randint(0, self.num_stimuli, (B, 2))
#         stim1 = torch.repeat_interleave(torch.scatter(torch.zeros(B, self.num_stimuli), 1, stimuli[:, 0].unsqueeze(1), 1), self.D, dim=1)
#         stim2 = torch.repeat_interleave(torch.scatter(torch.zeros(B, self.num_stimuli), 1, stimuli[:, 1].unsqueeze(1), 1), self.D, dim=1)
#         X = torch.cat(
#             [
#                 stim1.unsqueeze(1).expand(-1, self.stim_time, -1),
#                 torch.zeros((B, self.delay_time, self.D * self.num_stimuli)),
#                 stim2.unsqueeze(1).expand(-1, self.stim_time, -1),
#                 torch.zeros((B, self.decision_time, self.D * self.num_stimuli)),
#             ],
#             dim=1,
#         )
#         labels = (stimuli[:, 0] == stimuli[:, 1]).long()
#         params = dict(
#             stimuli=stimuli,
#         )
#         if self.as_spikes:
#             X = spikegen.rate(X, num_steps=1, gain=self.strength).squeeze(0)
#         return X, labels, params
