use fraction::Decimal;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct Weightjoint {
    pub position: u8,
    pub content: Option<Vec<Weightjoint>>, 
    pub content_value: Decimal,
    pub content_count: u32,
    pub accumulated_value: Decimal
}

#[derive(Debug)]
pub struct Hyperweight {
    pub content: Option<Vec<Weightjoint>>, 
    pub content_count: u32,
    pub accumulated_value: Decimal
}
impl Hyperweight {
    pub fn new() -> Self {
        Self {
            content: Some(vec![Weightjoint {position: 0, content: None, content_value: Decimal::from(0), content_count: 0, accumulated_value: Decimal::from(0)}; 10]),
            content_count: 0,
            accumulated_value: Decimal::from(0)
        }
    }
    pub fn get_point(position: u8, d: Decimal) -> usize {
        let m = Decimal::from(u32::pow(10, position.into()));
        ((m * d).floor() % 10).try_into().unwrap()
    }
    pub fn add_recurse(weightjoint: &mut Weightjoint, position: u8, weight: Decimal) {
        if position < weight.get_precision() {
            println!("{:?}", weightjoint);
        }
    }
    pub fn add(&mut self, weight: Decimal) {
        let point = Hyperweight::get_point(1, weight);
        println!("Point: {}", point);
        let weightjoint = self.content.as_mut().expect("Point is moot").get_mut(point).unwrap();
        self.content_count = self.content_count + 1;
        self.accumulated_value = self.accumulated_value + weight;
        Hyperweight::add_recurse(weightjoint, 1, weight);
    }
}

fn main() {
    let mut hw = Hyperweight::new();
    hw.add(Decimal::from(0.54321));
    println!("Count: {}", hw.content_count);
    println!("Accum: {}", hw.accumulated_value);
}
