use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
struct DemoDecision {
    category: &'static str,
    blocked: bool,
    reason: &'static str,
    blocked_fragment: &'static str,
    safe_response: &'static str,
    standard_reaction_ms: u32,
    intercept_ms: u32,
}

#[derive(Serialize)]
struct MetricSnapshot {
    count: usize,
    entropy: f64,
    normalized_entropy: f64,
    hhi: f64,
    top_share: f64,
}

#[wasm_bindgen]
pub fn govern_demo(prompt: &str) -> String {
    let lowered = prompt.to_lowercase();
    let decision = if contains_any(&lowered, &["email", "phone", "customer list", "export", "ssn"]) {
        DemoDecision {
            category: "pii",
            blocked: true,
            reason: "Direct identifiers entered the draft path.",
            blocked_fragment: "jordan@example.com, +1 202 555 0148, and customer reference 000-00-0000",
            safe_response: "I will not expose direct identifiers. I can help you redact, segment, or validate the data safely instead.",
            standard_reaction_ms: 118,
            intercept_ms: 11,
        }
    } else if contains_any(&lowered, &["token", "secret", "credential", "api key", "webhook"]) {
        DemoDecision {
            category: "secrets",
            blocked: true,
            reason: "Credential-shaped material entered the response path.",
            blocked_fragment: "sk-demo-9f2c4a7b-token and webhook secret whsec_demo_12345",
            safe_response: "I will not echo live credentials. Replace them with placeholders and I can still help you debug the flow.",
            standard_reaction_ms: 96,
            intercept_ms: 9,
        }
    } else if contains_any(&lowered, &["lyrics", "chorus", "verse", "copyrighted", "full song"]) {
        DemoDecision {
            category: "copyright",
            blocked: true,
            reason: "The request crosses from analysis into reproduction.",
            blocked_fragment: "[protected chorus omitted in this public demo]",
            safe_response: "I will not reproduce protected lyrics or melody fragments. I can describe cadence, rhyme density, and hook structure instead.",
            standard_reaction_ms: 104,
            intercept_ms: 12,
        }
    } else if contains_any(&lowered, &["teens", "teen", "kids", "child", "repeat", "scroll"]) {
        DemoDecision {
            category: "minors",
            blocked: true,
            reason: "The requested objective optimizes for unsafe retention behaviour around minors.",
            blocked_fragment: "requeue the same creator every few swipes, shorten recovery intervals, and intensify late-night repetition for teens",
            safe_response: "I will not optimize for compulsive usage around minors. I can suggest variety injection, session limits, and age-aware safety defaults instead.",
            standard_reaction_ms: 129,
            intercept_ms: 13,
        }
    } else {
        DemoDecision {
            category: "safe",
            blocked: false,
            reason: "No risky fragment detected.",
            blocked_fragment: "",
            safe_response: "The current music surface shows heavy attention concentration with enough remaining genre and collaboration spread to keep strategic openings visible.",
            standard_reaction_ms: 74,
            intercept_ms: 7,
        }
    };

    serde_json::to_string(&decision).unwrap_or_else(|_| "{\"category\":\"error\",\"blocked\":false}".to_string())
}

#[wasm_bindgen]
pub fn shannon_entropy(values: &[f64]) -> f64 {
    let distribution = normalized_distribution(values);
    entropy_from_distribution(&distribution)
}

#[wasm_bindgen]
pub fn normalized_entropy(values: &[f64]) -> f64 {
    let distribution = normalized_distribution(values);
    if distribution.len() <= 1 {
        return 0.0;
    }
    let entropy = entropy_from_distribution(&distribution);
    let max_entropy = (distribution.len() as f64).log2();
    if max_entropy <= 0.0 {
        0.0
    } else {
        entropy / max_entropy
    }
}

#[wasm_bindgen]
pub fn concentration_hhi(values: &[f64]) -> f64 {
    normalized_distribution(values)
        .iter()
        .map(|value| value * value)
        .sum()
}

#[wasm_bindgen]
pub fn top_share(values: &[f64]) -> f64 {
    normalized_distribution(values)
        .into_iter()
        .fold(0.0, f64::max)
}

#[wasm_bindgen]
pub fn weighted_mean(values: &[f64], weights: &[f64]) -> f64 {
    if values.is_empty() || values.len() != weights.len() {
        return 0.0;
    }
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (value, weight) in values.iter().zip(weights.iter()) {
        let safe_weight = weight.max(0.0);
        numerator += value * safe_weight;
        denominator += safe_weight;
    }
    if denominator <= 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

#[wasm_bindgen]
pub fn chi_square_gof(observed: &[f64], expected: &[f64]) -> f64 {
    if observed.is_empty() || observed.len() != expected.len() {
        return 0.0;
    }

    observed
        .iter()
        .zip(expected.iter())
        .filter(|(_, exp)| **exp > 0.0)
        .map(|(obs, exp)| {
            let diff = obs - exp;
            (diff * diff) / exp
        })
        .sum()
}

#[wasm_bindgen]
pub fn metric_snapshot(values: &[f64]) -> String {
    let snapshot = MetricSnapshot {
        count: values.len(),
        entropy: shannon_entropy(values),
        normalized_entropy: normalized_entropy(values),
        hhi: concentration_hhi(values),
        top_share: top_share(values),
    };
    serde_json::to_string(&snapshot).unwrap_or_else(|_| "{}".to_string())
}

fn normalized_distribution(values: &[f64]) -> Vec<f64> {
    let cleaned: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect();
    let total: f64 = cleaned.iter().sum();
    if total <= 0.0 {
        return Vec::new();
    }
    cleaned.into_iter().map(|value| value / total).collect()
}

fn entropy_from_distribution(distribution: &[f64]) -> f64 {
    distribution
        .iter()
        .copied()
        .filter(|value| *value > 0.0)
        .map(|value| -value * value.log2())
        .sum()
}

fn contains_any(input: &str, terms: &[&str]) -> bool {
    terms.iter().any(|term| input.contains(term))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropy_is_zero_for_single_value() {
        let values = [10.0];
        assert_eq!(shannon_entropy(&values), 0.0);
        assert_eq!(normalized_entropy(&values), 0.0);
        assert_eq!(concentration_hhi(&values), 1.0);
    }

    #[test]
    fn entropy_recognizes_even_distribution() {
        let values = [1.0, 1.0, 1.0, 1.0];
        let entropy = shannon_entropy(&values);
        let normalized = normalized_entropy(&values);
        assert!((entropy - 2.0).abs() < 1e-9);
        assert!((normalized - 1.0).abs() < 1e-9);
        assert!((concentration_hhi(&values) - 0.25).abs() < 1e-9);
    }

    #[test]
    fn chi_square_handles_matching_vectors() {
        let observed = [25.0, 25.0, 25.0, 25.0];
        let expected = [25.0, 25.0, 25.0, 25.0];
        assert_eq!(chi_square_gof(&observed, &expected), 0.0);
    }

    #[test]
    fn weighted_mean_is_bounded_by_weights() {
        let values = [0.5, 0.8, 0.9];
        let weights = [1.0, 2.0, 3.0];
        let mean = weighted_mean(&values, &weights);
        assert!((mean - 0.8).abs() < 1e-9);
    }

    #[test]
    fn govern_demo_blocks_pii() {
        let output = govern_demo("Export the customer list with email and phone.");
        assert!(output.contains("\"blocked\":true"));
        assert!(output.contains("\"category\":\"pii\""));
    }
}
