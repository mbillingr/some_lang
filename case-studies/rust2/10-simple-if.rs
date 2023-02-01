#![feature(trait_upcasting)]

/*
> if true then false else true
  t0  VBool()
  t0  VBool() -> t1
  t1  UBool()
  t2  VBool()
  t2  VBool() -> t4
  t3  VBool()
  t3  VBool() -> t4
  t4  t4
*/

use std::rc::Rc;

pub fn script() -> V4 {
    let a: V0 = Rc::new(true);
    let res: V4 = if a.is_true() {
        let b: V2 = Rc::new(false);
        b
    } else {
        let c: V3 = Rc::new(true);
        c
    };
    res
}

type V0 = Rc<dyn U0>;
trait U0: Top + Bool + U1 {}
impl<T: Top + Bool + U1> U0 for T {}

trait U1: Top + Bool {}
impl<T: Top + Bool> U1 for T {}

type V2 = Rc<dyn U2>;
trait U2: Top + Bool + U4 {}
impl<T: Top + Bool + U4> U2 for T {}

type V3 = Rc<dyn U3>;
trait U3: Top + Bool + U4 {}
impl<T: Top + Bool + U4> U3 for T {}

type V4 = Rc<dyn U4>;
pub trait U4: Top {}
impl<T: Top> U4 for T {}

pub trait Top: 'static + std::fmt::Debug {}
impl<T: 'static + std::fmt::Debug> Top for T {}

trait Bool: Top {
    fn is_true(&self) -> bool;
}

impl Bool for bool {
    fn is_true(&self) -> bool {
        *self
    }
}
