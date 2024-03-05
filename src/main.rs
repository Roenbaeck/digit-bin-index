use std::collections::HashMap;
use fraction::Decimal;
use rand::Rng;

#[derive(Debug)]
pub struct Weightjoint {
    pub position: u8,
    pub content: Option<Vec<Weightjoint>>, 
    pub content_value: Decimal,
    pub content_count: u32,
    pub accumulated_value: Decimal
}

#[derive(Debug)]
pub struct Hyperweight {
    pub content: HashMap<usize, Vec<Weightjoint>>, 
    pub content_count: u32,
    pub accumulated_value: Decimal
}
impl Hyperweight {
    pub fn new() -> Self {
        Self {
            content: HashMap::new(), 
            content_count: 0,
            accumulated_value: Decimal::from(0)
        }
    }
    pub fn get_point(position: u8, d: Decimal) -> usize {
        let m = Decimal::from(u32::pow(10, position.into()));
        ((m * d).floor() % 10).try_into().unwrap()
    }
    pub fn add_recurse(&mut self, Weightjoint: Weightjoint, position: u8, d: Decimal) {
        
    }
    pub fn add(&mut self, d: Decimal) {
        let mut point = Hyperweight::get_point(0, d);
        if self.content.get(&point).is_none() {
            self.content.insert(point, Vec::with_capacity(10));
        }
        let mut hypercontent = self.content.get_mut(&point).unwrap();
        for i in 1..d.get_precision() {
            point = Hyperweight::get_point(i, d);
            if hypercontent.get(point).is_none() {
                hypercontent[point] = Weightjoint {
                    position: i, 
                    content: None, 
                    content_value: Decimal::from(0), 
                    content_count: 0, 
                    accumulated_value: Decimal::from(0)
                }
            }
            hypercontent = hypercontent.get_mut(point).unwrap().content.unwrap();
        }
    }
}

fn main() {
    let mut hb = Hyperweight::new();
    hb.add(Decimal::from(9.12345));
    println!("{:?}", hb);
}
