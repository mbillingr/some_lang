#![feature(trait_upcasting)]

/*
> {a=true;b=false}
  t0  VBool()
  t1  VBool()
  t2  VObj(fields=AssocItem(key='b', val=1, next=AssocItem(key='a', val=0, next=AssocEmpty())))
*/

use std::rc::Rc;

fn main() {
    println!("{:?}", script());
}

fn script() -> V2 {
    Rc::new(R2 { a: Rc::new(true), b: Rc::new(false) } )
}

type V0 = Rc<dyn U0>;
trait U0: Top + Bool {}
impl<T: Top + Bool> U0 for T {}

type V1 = Rc<dyn U1>;
trait U1: Bool {}
impl<T: Top + Bool> U1 for T {}

type V2 = Rc<dyn U2>;
trait U2: Obj + Hasa<V0> + Hasb<V1> {}
impl<T: Top + Obj + Hasa<V0> + Hasb<V1>> U2 for T {}

#[derive(Debug)]
struct R2 { a: V0, b: V1 }
impl Obj for R2 {}
impl Hasa<V0> for R2 { fn get_a(&self) -> V0 { self.a.clone() } }
impl Hasb<V1> for R2 { fn get_b(&self) -> V1 { self.b.clone() } }


trait Hasa<T> {
    fn get_a(&self) -> T;
}


trait Hasb<T> {
    fn get_b(&self) -> T;
}

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
