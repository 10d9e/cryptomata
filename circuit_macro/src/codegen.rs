use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use std::collections::HashSet;
use syn::{parse_macro_input, FnArg, ItemFn, Pat, PatType};

use crate::expression::modify_body;

/// Generates the macro code based on the mode (either "compile" or "execute")
pub fn generate_macro(item: TokenStream, mode: &str) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    let fn_name = &input_fn.sig.ident;
    let inputs = &input_fn.sig.inputs;

    // get the type of the first input parameter
    let type_name = if let FnArg::Typed(PatType { ty, .. }) = &inputs[0] {
        quote! {#ty}
    } else {
        panic!("Expected typed argument");
    };

    // get the type of the first output parameter
    let output_type = if let syn::ReturnType::Type(_, ty) = &input_fn.sig.output {
        quote! {#ty}
    } else {
        panic!("Expected typed return type");
    };

    // We need to extract each input's identifier
    let mapped_inputs = inputs.iter().map(|input| {
        if let FnArg::Typed(PatType { pat, .. }) = input {
            if let Pat::Ident(pat_ident) = &**pat {
                let var_name = &pat_ident.ident;
                quote! {
                    let #var_name = &context.input(&#var_name.clone().into());
                }
            } else {
                quote! {}
            }
        } else {
            quote! {}
        }
    });

    // Extract constants to be added at the top of the function
    let mut constants = vec![];
    let transformed_block = modify_body(*input_fn.block, &mut constants);

    // remove duplicates
    let mut seen = HashSet::new();
    let constants: Vec<TokenStream2> = constants
        .into_iter()
        .filter(|item| seen.insert(item.to_string()))
        .collect();

    // Collect parameter names dynamically
    let param_names: Vec<_> = inputs
        .iter()
        .map(|input| {
            if let FnArg::Typed(PatType { pat, .. }) = input {
                if let Pat::Ident(pat_ident) = &**pat {
                    pat_ident.ident.clone()
                } else {
                    panic!("Expected identifier pattern");
                }
            } else {
                panic!("Expected typed argument");
            }
        })
        .collect();

    // Dynamically generate the `generate` function calls using the parameter names
    let match_arms = quote! {
        match std::any::type_name::<#type_name>() {
            "bool" => generate::<1, #type_name>(#(#param_names),*),
            "u8" => generate::<8, #type_name>(#(#param_names),*),
            "u16" => generate::<16, #type_name>(#(#param_names),*),
            "u32" => generate::<32, #type_name>(#(#param_names),*),
            "u64" => generate::<64, #type_name>(#(#param_names),*),
            "u128" => generate::<128, #type_name>(#(#param_names),*),
            _ => panic!("Unsupported type"),
        }
    };

    // Set the output type and operation logic based on mode
    let output_type = if mode == "compile" {
        quote! {(Circuit, Vec<bool>)}
    } else {
        quote! {#output_type}
    };

    let operation = if mode == "compile" {
        quote! {
            (context.compile(&output), context.inputs().to_vec())
        }
    } else {
        quote! {
            let compiled_circuit = context.compile(&output.into());
            let result = context.execute::<N>(&compiled_circuit).expect("Execution failed");
            result.into()
        }
    };

    // Build the function body with circuit context, compile, and execute
    let expanded = quote! {
        #[allow(non_camel_case_types, non_snake_case, clippy::builtin_type_shadow, unused_assignments)]
        fn #fn_name<#type_name>(#inputs) -> #output_type
        where
        #type_name: Into<GarbledUint<1>> + From<GarbledUint<1>>
                + Into<GarbledUint<8>> + From<GarbledUint<8>>
                + Into<GarbledUint<16>> + From<GarbledUint<16>>
                + Into<GarbledUint<32>> + From<GarbledUint<32>>
                + Into<GarbledUint<64>> + From<GarbledUint<64>>
                + Into<GarbledUint<128>> + From<GarbledUint<128>>
                + Clone,
        {
            fn generate<const N: usize, #type_name>(#inputs) -> #output_type
            where
                #type_name: Into<GarbledUint<N>> + From<GarbledUint<N>> + Clone,
            {
                let mut context = WRK17CircuitBuilder::default();
                #(#mapped_inputs)*
                #(#constants)*
                let const_true = &context.input::<N>(&true.into());
                let const_false = &context.input::<N>(&false.into());

                // Use the transformed function block (with context.add and if/else replacements)
                let output = { #transformed_block };

                #operation
            }

            #match_arms
        }
    };

    TokenStream::from(expanded)
}