use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::Hash;

pub struct DenseSet<T: Hash + Clone + Eq>
{
    element_to_idx_map: HashMap<T, usize>,
    dense_elements:     Vec<T>,
    changed_indices:    Option<Vec<usize>>
}

impl<T: Hash + Clone + Eq> DenseSet<T>
{
    #[allow(clippy::new_without_default)]
    pub fn new() -> DenseSet<T>
    {
        DenseSet {
            element_to_idx_map: HashMap::new(),
            dense_elements:     Vec::new(),
            changed_indices:    None
        }
    }

    pub fn new_index_tracking() -> DenseSet<T>
    {
        DenseSet {
            element_to_idx_map: HashMap::new(),
            dense_elements:     Vec::new(),
            changed_indices:    Some(Vec::new())
        }
    }

    /// Returns the previous element (if contained)
    pub fn insert(&mut self, t: T) -> Option<T>
    {
        match self.element_to_idx_map.entry(t.clone())
        {
            Entry::Occupied(occupied_entry) =>
            {
                let idx: usize = *occupied_entry.get();
                let old_t: T = occupied_entry.replace_key();

                self.dense_elements[idx] = t.clone();
                if let Some(v) = self.changed_indices.as_mut()
                {
                    v.push(idx)
                }

                Some(old_t)
            }
            Entry::Vacant(vacant_entry) =>
            {
                let new_element_idx = self.dense_elements.len();
                self.dense_elements.push(t.clone());
                if let Some(v) = self.changed_indices.as_mut()
                {
                    v.push(new_element_idx)
                }

                vacant_entry.insert(new_element_idx);

                None
            }
        }
    }

    pub fn remove(&mut self, t: T) -> Result<(), NoElementContained>
    {
        match self.element_to_idx_map.entry(t.clone())
        {
            Entry::Occupied(occupied_entry) =>
            {
                let (_, rem_idx) = occupied_entry.remove_entry();

                let last_idx = self.dense_elements.len() - 1;

                if rem_idx != last_idx
                {
                    // swap
                    self.dense_elements.swap(rem_idx, last_idx);

                    // fixup
                    *self
                        .element_to_idx_map
                        .get_mut(&self.dense_elements[rem_idx])
                        .unwrap() = rem_idx;
                }

                self.dense_elements.pop();
                if let Some(v) = self.changed_indices.as_mut()
                {
                    v.push(rem_idx);
                    v.push(last_idx)
                }

                Ok(())
            }
            Entry::Vacant(_) => Err(NoElementContained)
        }
    }

    pub fn to_dense_elements(&self) -> &[T]
    {
        &self.dense_elements
    }

    pub fn take_changed_indices(&mut self) -> Vec<usize>
    {
        std::mem::take(
            self.changed_indices
                .as_mut()
                .expect("Called take_change_indices on a DenseSet without index tracking enabled!")
        )
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct NoElementContained;

#[cfg(test)]
mod tests
{
    use super::*;

    #[test]
    fn test_new_dense_set_is_empty()
    {
        let set: DenseSet<i32> = DenseSet::new();
        assert!(set.to_dense_elements().is_empty());
    }

    #[test]
    fn test_insert_element()
    {
        let mut set = DenseSet::new();
        assert_eq!(set.insert(1), None);
        assert_eq!(set.to_dense_elements(), &[1]);
    }

    #[test]
    fn test_insert_duplicate_element()
    {
        let mut set = DenseSet::new();
        assert_eq!(set.insert(1), None);
        assert_eq!(set.insert(1), Some(1));
        assert_eq!(set.to_dense_elements(), &[1]);
    }

    #[test]
    fn test_insert_multiple_elements()
    {
        let mut set = DenseSet::new();
        assert_eq!(set.insert(1), None);
        assert_eq!(set.insert(2), None);
        assert_eq!(set.to_dense_elements(), &[1, 2]);
    }

    #[test]
    fn test_remove_existing_element()
    {
        let mut set = DenseSet::new();
        set.insert(1);
        set.insert(2);
        assert_eq!(set.remove(1), Ok(()));
        assert_eq!(set.to_dense_elements(), &[2]);
    }

    #[test]
    fn test_remove_non_existing_element()
    {
        let mut set = DenseSet::new();
        set.insert(1);
        assert_eq!(set.remove(2), Err(NoElementContained));
        assert_eq!(set.to_dense_elements(), &[1]);
    }

    #[test]
    fn test_remove_last_element()
    {
        let mut set = DenseSet::new();
        set.insert(1);
        assert_eq!(set.remove(1), Ok(()));
        assert!(set.to_dense_elements().is_empty());
    }

    #[test]
    fn test_remove_and_insert_element()
    {
        let mut set = DenseSet::new();
        set.insert(1);
        set.insert(2);
        set.remove(1).unwrap();
        assert_eq!(set.insert(1), None);
        assert_eq!(set.to_dense_elements(), &[2, 1]);
    }

    #[test]
    fn test_insert_after_removal()
    {
        let mut set = DenseSet::new();
        set.insert(1);
        set.remove(1).unwrap();
        assert_eq!(set.insert(1), None);
        assert_eq!(set.to_dense_elements(), &[1]);
    }

    #[test]
    fn test_remove_swapped_element()
    {
        let mut set = DenseSet::new();
        set.insert(1);
        set.insert(2);
        set.insert(3);
        set.remove(2).unwrap();
        assert_eq!(set.to_dense_elements(), &[1, 3]);
        set.remove(3).unwrap();
        assert_eq!(set.to_dense_elements(), &[1]);
    }

    #[test]
    fn test_remove_element_from_empty_set()
    {
        let mut set: DenseSet<i32> = DenseSet::new();
        assert_eq!(set.remove(1), Err(NoElementContained));
    }

    #[test]
    fn test_insert_large_number_of_elements()
    {
        let mut set = DenseSet::new();
        for i in 0..1000
        {
            assert_eq!(set.insert(i), None);
        }
        let elements: Vec<i32> = (0..1000).collect();
        assert_eq!(set.to_dense_elements(), elements.as_slice());
    }

    #[test]
    fn test_remove_all_elements()
    {
        let mut set = DenseSet::new();
        for i in 0..100
        {
            set.insert(i);
        }
        for i in 0..100
        {
            assert_eq!(set.remove(i), Ok(()));
        }
        assert!(set.to_dense_elements().is_empty());
    }

    #[test]
    fn test_remove_elements_out_of_order()
    {
        let mut set = DenseSet::new();
        set.insert(1);
        set.insert(2);
        set.insert(3);
        assert_eq!(set.remove(2), Ok(()));
        assert_eq!(set.to_dense_elements(), &[1, 3]);
        assert_eq!(set.remove(1), Ok(()));
        assert_eq!(set.to_dense_elements(), &[3]);
        assert_eq!(set.remove(3), Ok(()));
        assert!(set.to_dense_elements().is_empty());
    }

    #[test]
    fn test_reinsertion_of_removed_element()
    {
        let mut set = DenseSet::new();
        set.insert(1);
        set.remove(1).unwrap();
        assert_eq!(set.insert(1), None);
        assert_eq!(set.to_dense_elements(), &[1]);
    }

    #[test]
    fn test_insert_after_clear()
    {
        let mut set = DenseSet::new();
        set.insert(1);
        set.insert(2);
        set.remove(1).unwrap();
        set.remove(2).unwrap();
        assert!(set.to_dense_elements().is_empty());
        assert_eq!(set.insert(3), None);
        assert_eq!(set.to_dense_elements(), &[3]);
    }

    #[test]
    fn test_remove_and_insert_same_element()
    {
        let mut set = DenseSet::new();
        set.insert(1);
        set.remove(1).unwrap();
        assert_eq!(set.insert(1), None);
        assert_eq!(set.to_dense_elements(), &[1]);
    }

    #[test]
    fn test_remove_middle_element()
    {
        let mut set = DenseSet::new();
        set.insert(1);
        set.insert(2);
        set.insert(3);
        set.remove(2).unwrap();
        assert_eq!(set.to_dense_elements(), &[1, 3]);
    }

    #[test]
    fn test_large_number_of_insertions_and_removals()
    {
        let mut set = DenseSet::new();
        for i in 0..1000
        {
            set.insert(i);
        }
        for i in 0..1000
        {
            set.remove(i).unwrap();
        }
        assert!(set.to_dense_elements().is_empty());
    }
}
