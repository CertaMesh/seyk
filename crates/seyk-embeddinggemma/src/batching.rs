use crate::{Error, Result};
use std::ops::Range;

/// Per-call padded work limits, applied independently to each text session.
/// These bound tensor shapes, not process memory or tokenizer queues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchBudget {
    pub padded_tokens: usize,
    pub attention_cells: usize,
}

impl BatchBudget {
    /// Candidate limits for measurement: 16 rows of 512 tokens.
    pub const CANDIDATE: Self = Self {
        padded_tokens: 8192,
        attention_cells: 4_194_304,
    };

    fn fits(self, rows: usize, length: usize) -> bool {
        rows.checked_mul(length).is_some_and(|tokens| {
            tokens <= self.padded_tokens
                && tokens
                    .checked_mul(length)
                    .is_some_and(|cells| cells <= self.attention_cells)
        })
    }

    /// Check that both limits are nonzero and can admit a maximum-length singleton.
    pub fn validate(self, max_tokens: usize) -> Result<()> {
        if self.padded_tokens == 0 || self.attention_cells == 0 || !self.fits(1, max_tokens) {
            return Err(Error::Config(
                "batch budgets must fit a max_tokens singleton".into(),
            ));
        }
        Ok(())
    }
}

pub(crate) struct BatchPlan {
    pub order: Vec<usize>,
    pub ranges: Vec<Range<usize>>,
}

pub(crate) fn plan(
    lengths: &[usize],
    row_cap: usize,
    budget: Option<BatchBudget>,
) -> Result<BatchPlan> {
    if row_cap == 0 {
        return Err(Error::Config("batch_size must be nonzero".into()));
    }
    if let Some(budget) = budget {
        budget.validate(lengths.iter().copied().max().unwrap_or(0))?;
    }
    let mut order: Vec<usize> = (0..lengths.len()).collect();
    order.sort_by_key(|&i| lengths[i]);
    let mut ranges = Vec::new();
    let mut start = 0;
    for (position, &index) in order.iter().enumerate() {
        let rows = position - start + 1;
        if rows > row_cap || budget.is_some_and(|b| !b.fits(rows, lengths[index])) {
            ranges.push(start..position);
            start = position;
        }
    }
    if start < order.len() {
        ranges.push(start..order.len());
    }
    Ok(BatchPlan { order, ranges })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_and_mixed_inputs_obey_padded_cost_and_preserve_every_row() {
        for lengths in [
            vec![1727; 39],
            vec![1598; 19],
            vec![1773; 39],
            vec![2048, 1, 512, 1024, 0, 512, 1700],
            vec![],
        ] {
            let b = BatchBudget::CANDIDATE;
            let plan = plan(&lengths, 16, Some(b)).unwrap();
            let mut seen = Vec::new();
            for range in plan.ranges {
                let indices = &plan.order[range];
                let max = indices.iter().map(|&i| lengths[i]).max().unwrap();
                assert!(indices.len() <= 16);
                assert!(b.fits(indices.len(), max));
                seen.extend_from_slice(indices);
            }
            seen.sort_unstable();
            assert_eq!(seen, (0..lengths.len()).collect::<Vec<_>>());
        }
    }

    #[test]
    fn ordinary_prose_and_boundaries_keep_full_batches() {
        let b = BatchBudget::CANDIDATE;
        assert_eq!(
            plan(&[512; 32], 16, Some(b)).unwrap().ranges,
            vec![0..16, 16..32]
        );
        assert_eq!(
            plan(&[1024; 5], 16, Some(b)).unwrap().ranges,
            vec![0..4, 4..5]
        );
        assert_eq!(
            plan(&[2048; 2], 16, Some(b)).unwrap().ranges,
            vec![0..1, 1..2]
        );
        let token_only = BatchBudget {
            padded_tokens: 1024,
            attention_cells: usize::MAX,
        };
        assert_eq!(
            plan(&[512; 3], 16, Some(token_only)).unwrap().ranges,
            vec![0..2, 2..3]
        );
        assert_eq!(
            plan(&[7, 1, 1, 7], 2, None).unwrap().order,
            vec![1, 2, 0, 3]
        );
    }

    #[test]
    fn invalid_and_overflowing_costs_fail_before_admission() {
        assert!(plan(&[1], 0, None).is_err());
        for b in [
            BatchBudget {
                padded_tokens: 0,
                attention_cells: 1,
            },
            BatchBudget {
                padded_tokens: 1,
                attention_cells: 0,
            },
            BatchBudget::CANDIDATE,
        ] {
            assert!(plan(&[2049], 16, Some(b)).is_err());
        }
        let b = BatchBudget {
            padded_tokens: usize::MAX,
            attention_cells: usize::MAX,
        };
        assert!(!b.fits(2, usize::MAX));
        assert!(plan(&[usize::MAX], 16, Some(b)).is_err());
    }
}
