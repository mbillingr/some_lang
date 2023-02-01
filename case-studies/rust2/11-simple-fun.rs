#![feature(trait_upcasting)]

/*
> fun x -> true
  t0  t0
  t1  VBool()
  t2  VFunc(arg=0, ret=1)
*/

use std::rc::Rc;

fn main() {
    println!("{:?}", script());
}

fn script() -> V2 {
    fun::<V0, V1, _>(|x| Rc::new(true) as V1)
}

type V0 = Rc<dyn U0>;
trait U0: Top {}
impl<T: Top> U0 for T {}

type V1 = Rc<dyn U1>;
trait U1: Top + Bool {}
impl<T: Top + Bool> U1 for T {}

type V2 = Rc<dyn U2>;
trait U2: Top + Fun<V0,V1> {}
impl<T: Top + Fun<V0,V1>> U2 for T {}


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

trait Fun<A,R>: Top {
    fn apply(&self, a: A) -> R;
}

fn fun<A, R, F>(f: F) -> Rc<Function<A, R, F>> {
    Rc::new(Function(f, std::marker::PhantomData))
}

struct Function<A, R, F>(F, std::marker::PhantomData<(A,R)>);

impl<A: Top, R: Top, F: 'static + Fn(A)->R> Fun<A, R> for Function<A, R, F> {
    fn apply(&self, a: A) -> R {
        (self.0)(a)
    }
}

impl<A, R, F> std::fmt::Debug for Function<A, R, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "<fun {} -> {}>", std::any::type_name::<A>(), std::any::type_name::<R>())
    }
}
