"""Score fusion module for combining acoustic and linguistic predictions."""


class ScoreFusion:
    """
    Weighted fusion of acoustic and linguistic scores.

    Default weights:
    - Acoustic: 0.6 (audio artifacts are more reliable)
    - Linguistic: 0.4 (text patterns can be misleading)
    """

    def __init__(self, acoustic_weight: float = 0.6, linguistic_weight: float = 0.4):
        """
        Initialize fusion module with weights.

        Args:
            acoustic_weight: Weight for acoustic score (0-1)
            linguistic_weight: Weight for linguistic score (0-1)

        Note: Weights should sum to 1.0 for proper probability interpretation
        """
        assert abs((acoustic_weight + linguistic_weight) - 1.0) < 1e-6, \
            "Weights must sum to 1.0"

        self.acoustic_weight = acoustic_weight
        self.linguistic_weight = linguistic_weight

    def fuse(self, acoustic_score: float, linguistic_score: float) -> float:
        """
        Compute weighted average of acoustic and linguistic scores.

        Args:
            acoustic_score: Probability from acoustic model (0-1)
            linguistic_score: Probability from linguistic model (0-1)

        Returns:
            Fused probability score (0-1)
        """
        fused_score = (
            self.acoustic_weight * acoustic_score +
            self.linguistic_weight * linguistic_score
        )
        return fused_score

    def decide(self, fused_score: float, threshold: float = 0.5) -> str:
        """
        Make binary decision based on fused score.

        Args:
            fused_score: Fused probability score (0-1)
            threshold: Decision threshold (default: 0.5)

        Returns:
            "FAKE" if score >= threshold, else "REAL"
        """
        return "FAKE" if fused_score >= threshold else "REAL"

    def get_confidence(self, fused_score: float) -> float:
        """
        Calculate confidence of prediction.

        Args:
            fused_score: Fused probability score (0-1)

        Returns:
            Confidence level (0-1), where 1.0 is most confident
        """
        # Distance from decision boundary (0.5)
        confidence = abs(fused_score - 0.5) * 2
        return confidence


class AdaptiveFusion(ScoreFusion):
    """
    Adaptive fusion that adjusts weights based on individual model confidence.

    If one model is much more confident than the other, give it more weight.
    """

    def __init__(self, base_acoustic_weight: float = 0.6, base_linguistic_weight: float = 0.4):
        super().__init__(base_acoustic_weight, base_linguistic_weight)
        self.base_acoustic_weight = base_acoustic_weight
        self.base_linguistic_weight = base_linguistic_weight

    def fuse_adaptive(self, acoustic_score: float, linguistic_score: float) -> tuple[float, dict]:
        """
        Compute adaptive weighted average based on individual confidences.

        Args:
            acoustic_score: Probability from acoustic model (0-1)
            linguistic_score: Probability from linguistic model (0-1)

        Returns:
            Tuple of (fused_score, weights_used)
        """
        # Calculate confidence for each model
        acoustic_conf = abs(acoustic_score - 0.5) * 2
        linguistic_conf = abs(linguistic_score - 0.5) * 2

        # If both models are uncertain, use base weights
        if acoustic_conf < 0.3 and linguistic_conf < 0.3:
            weights = {
                'acoustic': self.base_acoustic_weight,
                'linguistic': self.base_linguistic_weight
            }
        else:
            # Adjust weights based on relative confidence
            total_conf = acoustic_conf + linguistic_conf
            if total_conf > 0:
                acoustic_adjusted = (acoustic_conf / total_conf) * 0.8 + 0.1
                linguistic_adjusted = (linguistic_conf / total_conf) * 0.8 + 0.1

                # Normalize to sum to 1
                total = acoustic_adjusted + linguistic_adjusted
                weights = {
                    'acoustic': acoustic_adjusted / total,
                    'linguistic': linguistic_adjusted / total
                }
            else:
                weights = {
                    'acoustic': self.base_acoustic_weight,
                    'linguistic': self.base_linguistic_weight
                }

        # Compute fused score with adjusted weights
        fused_score = (
            weights['acoustic'] * acoustic_score +
            weights['linguistic'] * linguistic_score
        )

        return fused_score, weights


if __name__ == "__main__":
    # Test standard fusion
    fusion = ScoreFusion(acoustic_weight=0.6, linguistic_weight=0.4)

    test_cases = [
        (0.8, 0.7),  # Both say fake
        (0.2, 0.3),  # Both say real
        (0.9, 0.2),  # Acoustic says fake, linguistic says real
        (0.3, 0.8),  # Acoustic says real, linguistic says fake
    ]

    print("Standard Fusion:")
    for acoustic, linguistic in test_cases:
        fused = fusion.fuse(acoustic, linguistic)
        decision = fusion.decide(fused)
        confidence = fusion.get_confidence(fused)
        print(f"A:{acoustic:.2f} L:{linguistic:.2f} → Fused:{fused:.2f} → {decision} (conf:{confidence:.2f})")

    print("\nAdaptive Fusion:")
    adaptive = AdaptiveFusion()
    for acoustic, linguistic in test_cases:
        fused, weights = adaptive.fuse_adaptive(acoustic, linguistic)
        decision = adaptive.decide(fused)
        confidence = adaptive.get_confidence(fused)
        print(f"A:{acoustic:.2f} L:{linguistic:.2f} → Fused:{fused:.2f} → {decision} "
              f"(weights: A={weights['acoustic']:.2f}, L={weights['linguistic']:.2f})")
