use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Expr, LitBool, LitInt};

pub mod handle_constant {
    use super::*;

    pub fn bool_literal(lit_bool: LitBool, constants: &mut Vec<TokenStream>) -> Expr {
        let value = lit_bool.value;
        let const_var = format_ident!("const_{}", value as u128);

        if value {
            constants.push(quote! {
                let #const_var = &context.input::<N>(&1_u128.into()).clone();
            });
        } else {
            constants.push(quote! {
                let #const_var = &context.input::<N>(&0_u128.into()).clone();
            });
        }
        syn::parse_quote! {#const_var}
    }

    pub fn int_literal(lit_int: LitInt, constants: &mut Vec<TokenStream>) -> Expr {
        let value = lit_int
            .base10_parse::<u128>()
            .expect("Expected an integer literal");
        let const_var = format_ident!("const_{}", value);
        constants.push(quote! {
            let #const_var = &context.input::<N>(&#value.into()).clone();
        });
        syn::parse_quote! {#const_var}
    }
}