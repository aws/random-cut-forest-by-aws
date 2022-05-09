use std::fmt::Debug;

#[repr(C)]
pub struct IntervalStoreManager<T> {
    capacity: usize,
    last_in_use: usize,
    free_indices_start: Vec<T>,
    free_indices_end: Vec<T>,
}

impl<T: Copy + std::convert::TryFrom<usize>> IntervalStoreManager<T>
where
    T: std::fmt::Display + std::cmp::PartialEq,
    usize: From<T>,
{
    pub fn new(size: usize) -> Self
    where
        <T as TryFrom<usize>>::Error: Debug,
    {
        IntervalStoreManager {
            capacity: size,
            last_in_use: 1,
            free_indices_start: vec![0.try_into().unwrap()],
            free_indices_end: vec![(size - 1).try_into().unwrap()],
        }
    }

    pub fn get_capacity(&self) -> usize {
        self.capacity
    }

    pub fn change_capacity(&mut self, new_capacity: usize)
    where
        <T as TryFrom<usize>>::Error: Debug,
    {
        if new_capacity > self.capacity {
            let start: T = self.capacity.try_into().unwrap();
            let end: T = (new_capacity - 1).try_into().unwrap();
            if self.free_indices_start.len() == self.last_in_use {
                self.free_indices_start.resize(self.last_in_use + 1, start);
                self.free_indices_end.resize(self.last_in_use + 1, end);
            } else {
                self.free_indices_start[self.last_in_use] = start;
                self.free_indices_end[self.last_in_use] = end;
            }
            self.last_in_use += 1;
            self.capacity = new_capacity;
        }
    }

    pub fn is_empty(&self) -> bool {
        self.last_in_use == 0
    }

    pub fn get(&mut self) -> usize
    where
        <T as TryFrom<usize>>::Error: Debug,
    {
        if self.is_empty() {
            println!(" no more indices left");
            panic!();
        }
        let answer = self.free_indices_start[self.last_in_use - 1];
        let new_value: usize = answer.into();
        if answer == self.free_indices_end[self.last_in_use - 1] {
            self.last_in_use -= 1;
        } else {
            self.free_indices_start[self.last_in_use - 1] = (new_value + 1).try_into().unwrap();
        }
        new_value
    }

    pub fn release(&mut self, index: usize)
    where
        <T as TryFrom<usize>>::Error: Debug,
    {
        let val: T = TryFrom::try_from(index).unwrap();
        if self.last_in_use != 0 {
            let start: usize = self.free_indices_start[self.last_in_use - 1].into();
            let end: usize = self.free_indices_end[self.last_in_use - 1].into();
            if start == index + 1 {
                self.free_indices_start[self.last_in_use - 1] = val;
                return;
            } else if end + 1 == index {
                self.free_indices_end[self.last_in_use - 1] = val;
                return;
            }
        }
        if self.last_in_use < self.free_indices_start.len() {
            self.free_indices_start[self.last_in_use] = val;
            self.free_indices_end[self.last_in_use] = val;
        } else {
            self.free_indices_start.resize(self.last_in_use + 1, val);
            self.free_indices_end.resize(self.last_in_use + 1, val);
        }
        self.last_in_use += 1;
    }

    pub fn used(&self) -> usize {
        let mut answer = 0;
        for i in 0..self.last_in_use {
            let start: usize = self.free_indices_start[i].into();
            let end: usize = self.free_indices_end[i].into();
            answer += end - start + 1;
        }
        self.capacity - answer
    }

    pub fn get_size(&self) -> usize {
        self.free_indices_start.len() * 2 * std::mem::size_of::<T>()
            + std::mem::size_of::<IntervalStoreManager<T>>()
    }
}
