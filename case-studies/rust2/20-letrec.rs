#![feature(trait_upcasting)]


use std::rc::Rc;

fn main() {
    println!("{:?}", script());
}

fn script() -> V1 {
    let cls = Rc::new(letrec123::Closure {});
    let even = {
        let cls = cls.clone();
        fun::<V1, V1, _>(move |x| letrec123::even(x, &*cls))
    };
    let odd = {
        let cls = cls.clone();
        fun::<V1, V1, _>(move |x| letrec123::odd(x, &*cls))
    };

    even.apply(Rc::new(false))
}

mod letrec123 {
    use super::*;

    pub struct Closure {}

    pub fn even(x: V1, cls: &Closure) -> V1 {
        Rc::new(!odd(x, cls).is_true())
    }

    pub fn odd(x: V1, cls: &Closure) -> V1 {
        x
    }
}

type V1 = Rc<dyn U1>;
pub trait U1: Top + Bool {}
impl<T: Top + Bool> U1 for T {}


// common code

pub trait Top: 'static + std::fmt::Debug {}
impl<T: 'static + std::fmt::Debug> Top for T {}

pub trait Bool: Top {
    fn is_true(&self) -> bool;
}

impl Bool for bool {
    fn is_true(&self) -> bool {
        *self
    }
}

trait Fun<A, R>: Top {
    fn apply(&self, a: A) -> R;
}

fn fun<A, R, F>(f: F) -> Rc<Function<A, R, F>> {
    Rc::new(Function(f, std::marker::PhantomData))
}

struct Function<A, R, F>(F, std::marker::PhantomData<(A, R)>);

impl<A: Top, R: Top, F: 'static + Fn(A) -> R> Fun<A, R> for Function<A, R, F> {
    fn apply(&self, a: A) -> R {
        (self.0)(a)
    }
}

impl<A, R, F> std::fmt::Debug for Function<A, R, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "<fun {} -> {}>",
            std::any::type_name::<A>(),
            std::any::type_name::<R>()
        )
    }
}
