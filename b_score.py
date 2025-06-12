import math
from typing import List, Dict


K = 100
Z = 1.96
P_THRESHOLD = 0.1  # ratio for common opponents


def _wilson_lower_bound(wins: float, n: float, z: float = Z) -> float:
    """Return Wilson score lower bound for win probability."""
    if n <= 0:
        return 0.0
    phat = wins / n
    denominator = 1 + z**2 / n
    centre = phat + z**2 / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)
    return (centre - margin) / denominator


def calculate_b_score(data: List[Dict]) -> List[Dict]:
    """Calculate B-Score for each deck based on provided matchup data."""
    results: List[Dict[str, float]] = []
    for deck in data:
        total_matches = deck.get("total_matches", 0)
        opponents = deck.get("opponents", [])

        if not opponents or total_matches <= 0:
            results.append({"deck": deck.get("deck", ""), "B-Score": 0.0})
            continue

        # Determine common opponents by dynamic threshold
        threshold = max(10, round(total_matches * P_THRESHOLD))
        common = [o for o in opponents if o.get("games", 0) >= threshold]

        if not common:
            common = opponents

        # Step 1: weighted win rate over common opponents
        weighted_wins = sum(o["win_rate"] * o["games"] * o.get("weight", 1) for o in common)
        weighted_games = sum(o["games"] * o.get("weight", 1) for o in common)
        wr_w_raw = weighted_wins / weighted_games if weighted_games else 0.0

        # Step 2: Wilson lower bound for aggregated performance
        wr_wilson = _wilson_lower_bound(weighted_wins, weighted_games)

        # Step 3: baseline p0 using unweighted data
        total_wins = sum(o["win_rate"] * o["games"] for o in common)
        total_games = sum(o["games"] for o in common)
        p0 = total_wins / total_games if total_games else 0.0

        # Step 4: Bayesian smoothed win rate
        wr_b = (p0 * K + total_wins) / (K + total_games)

        # Step 5: alpha based on total matches
        alpha = total_matches / (total_matches + K)

        # Step 6: B-Score before deductions
        B = (1 - alpha) * wr_b + alpha * wr_wilson

        # Step 7: disadvantage deduction
        deduction = 0.0
        for opp in common:
            opp_games = opp["games"]
            opp_rate = opp["win_rate"]
            lower = _wilson_lower_bound(opp_rate * opp_games, opp_games)
            if lower < 0.30:
                deduction += opp_games / total_matches

        B_final = max(0.0, B - deduction)
        results.append({"deck": deck.get("deck", ""), "B-Score": round(B_final * 100, 1)})

    return results
