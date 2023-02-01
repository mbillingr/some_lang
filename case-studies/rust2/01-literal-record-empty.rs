#![feature(trait_upcasting)]

/*
> {}
  t0  VObj(fields=AssocEmpty())
*/

use std::rc::Rc;

fn script() -> V0 {
    Rc::new(R0 {})
}

type V0 = Rc<dyn U0>;
trait U0: Top + Obj {}
impl<T: Top + Obj> U0 for T {}

#[derive(Debug)]
struct R0 {}
impl Obj for R0 {}

// common code

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

trait Obj: Top {}
