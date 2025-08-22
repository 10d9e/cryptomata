extern crate proc_macro;

use proc_macro::TokenStream;
use syn::parse_macro_input;

mod codegen;
mod constants;
mod expression;

use codegen::generate_macro;

#[proc_macro_attribute]
pub fn encrypted(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mode = parse_macro_input!(attr as syn::Ident).to_string(); // Retrieve the mode (e.g., "compile" or "execute")
    generate_macro(item, &mode)
}
