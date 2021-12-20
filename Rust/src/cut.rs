/**
*  this is a class that helps manage the cut information; the nodes do not store information in
*  this format
*/


pub struct Cut {
    dimension: usize,
    value: f32,
}

impl Cut {
    pub fn new(dim :usize, value : f32) -> Self {
        Cut {
	    dimension : dim,
            value
        }
    }

    pub fn get_dimension(&self) -> usize {
        self.dimension
    }

    pub fn get_value(&self) -> f32 {
        self.value
    }
   
}
