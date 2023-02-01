#![feature(trait_upcasting)]

/*
> true
  t0  VBool()
*/

use std::rc::Rc;

fn script() -> V0 {
    Rc::new(true)
}

type V0 = Rc<dyn U0>;
trait U0: Top + Bool {}
impl<T: Top + Bool> U0 for T {}

trait Top: 'static + std::fmt::Debug {}
impl<T: 'static + std::fmt::Debug> Top for T {}

trait Bool: Top {
    fn is_true(&self) -> bool;
}

impl Bool for bool {
    fn is_true(&self) -> bool {
        *self
    }
}
