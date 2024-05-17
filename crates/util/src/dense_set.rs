use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::Hash;

pub struct DenseSet<T: Hash + Clone + Eq>
{
    element_to_idx_map: HashMap<T, usize>,
    dense_elements:     Vec<T>
}

impl<T: Hash + Clone + Eq> DenseSet<T>
{
    pub fn new() -> DenseSet<T>
    {
        DenseSet {
            element_to_idx_map: HashMap::new(),
            dense_elements:     Vec::new()
        }
    }

    /// Returns the previous element (if contained)
    #[must_use]
    pub fn insert(&mut self, t: T) -> Option<T>
    {
        match self.element_to_idx_map.entry(t.clone())
        {
            Entry::Occupied(occupied_entry) =>
            {
                let idx: usize = *occupied_entry.get();
                let old_t: T = occupied_entry.replace_key();

                self.dense_elements[idx] = t;

                Some(old_t)
            }
            Entry::Vacant(vacant_entry) =>
            {
                let new_element_idx = self.dense_elements.len();
                self.dense_elements.push(t);

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

                Ok(())
            }
            Entry::Vacant(_) => Err(NoElementContained)
        }
    }

    pub fn to_dense_elements(&self) -> &[T]
    {
        &self.dense_elements
    }
}

pub struct NoElementContained;
