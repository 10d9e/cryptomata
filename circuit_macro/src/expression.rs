use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    BinOp, Expr, ExprAssign, ExprBinary, ExprBlock, ExprIf, ExprLet, ExprMatch, ExprReference,
    ExprUnary, Lit, Pat,
};

use crate::constants::handle_constant;

/// Traverse and transform the function body, replacing binary operators and if/else expressions.
/// Also collects constants to add to the circuit context.
pub fn modify_body(block: syn::Block, constants: &mut Vec<TokenStream>) -> syn::Block {
    let stmts = block
        .stmts
        .into_iter()
        .map(|stmt| match stmt {
            syn::Stmt::Expr(expr, semi_opt) => {
                syn::Stmt::Expr(replace_expressions(expr, constants), semi_opt)
            }
            syn::Stmt::Local(mut local) => {
                if let Some(local_init) = &mut local.init {
                    let local_expr = replace_expressions(*local_init.expr.clone(), constants);

                    if let syn::Pat::Ident(ref pat_ident) = local.pat {
                        if pat_ident.mutability.is_some() {
                            local_init.expr = Box::new(syn::parse_quote! {
                                #local_expr.clone()
                            });
                        } else {
                            local_init.expr = Box::new(syn::parse_quote! {
                                #local_expr
                            });
                        }
                    }
                }
                syn::Stmt::Local(local)
            }

            other => other,
        })
        .collect();

    syn::Block {
        stmts,
        brace_token: syn::token::Brace::default(),
    }
}

/// Replaces binary operators and if/else expressions with appropriate context calls.
pub fn replace_expressions(expr: Expr, constants: &mut Vec<TokenStream>) -> Expr {
    match expr {
        // if there is a block, recursively call modify_body
        Expr::Block(ExprBlock { block, .. }) => {
            let transformed_block = modify_body(block, constants);
            syn::parse_quote! { #transformed_block }
        }
        // implement assignment
        Expr::Assign(ExprAssign { left, right, .. }) => {
            let left_expr = replace_expressions(*left, constants);
            let right_expr = replace_expressions(*right, constants);

            match right_expr {
                Expr::Reference(ExprReference { .. }) => {
                    syn::parse_quote! {
                        #left_expr = &#right_expr.clone()
                    }
                }
                _ => {
                    syn::parse_quote! {
                        #left_expr = #right_expr.clone()
                    }
                }
            }
        }
        // return statement
        Expr::Return(_) => {
            panic!("Return statement not allowed in circuit macro");
        }
        // parentheses to ensure proper order of operations
        Expr::Paren(expr_paren) => {
            let inner_expr = replace_expressions(*expr_paren.expr, constants);
            syn::parse_quote! { (#inner_expr) }
        }
        // boolean literal
        Expr::Lit(syn::ExprLit {
            lit: Lit::Bool(lit_bool),
            ..
        }) => handle_constant::bool_literal(lit_bool, constants),
        // integer literal - handle as a constant in the circuit context
        Expr::Lit(syn::ExprLit {
            lit: Lit::Int(lit_int),
            ..
        }) => handle_constant::int_literal(lit_int, constants),
        // Binary expressions
        Expr::Binary(expr_binary) => handle_binary_expression(expr_binary, constants),
        // Unary expressions
        Expr::Unary(expr_unary) => handle_unary_expression(expr_unary, constants),
        // If expressions
        Expr::If(expr_if) => handle_if_expression(expr_if, constants),
        // Match expressions
        Expr::Match(expr_match) => handle_match_expression(expr_match, constants),

        other => other,
    }
}

fn handle_binary_expression(expr: ExprBinary, constants: &mut Vec<TokenStream>) -> Expr {
    let ExprBinary { left, right, op, .. } = expr;

    match op {
        // Comparison operators
        BinOp::Eq(_) => {
            let left_expr = replace_expressions(*left, constants);
            let right_expr = replace_expressions(*right, constants);
            syn::parse_quote! {{
                let left = #left_expr;
                let right = #right_expr;
                context.eq(&left.into(), &right.into())
            }}
        }
        BinOp::Ne(_) => {
            let left_expr = replace_expressions(*left, constants);
            let right_expr = replace_expressions(*right, constants);
            syn::parse_quote! {{
                let left = #left_expr;
                let right = #right_expr;
                context.ne(&left.into(), &right.into())
            }}
        }
        BinOp::Gt(_) => {
            let left_expr = replace_expressions(*left, constants);
            let right_expr = replace_expressions(*right, constants);
            syn::parse_quote! {{
                let left = #left_expr;
                let right = #right_expr;
                context.gt(&left.into(), &right.into())
            }}
        }
        BinOp::Ge(_) => {
            let left_expr = replace_expressions(*left, constants);
            let right_expr = replace_expressions(*right, constants);
            syn::parse_quote! {{
                let left = #left_expr;
                let right = #right_expr;
                context.ge(&left.into(), &right.into())
            }}
        }
        BinOp::Lt(_) => {
            let left_expr = replace_expressions(*left, constants);
            let right_expr = replace_expressions(*right, constants);
            syn::parse_quote! {{
                let left = #left_expr;
                let right = #right_expr;
                context.lt(&left.into(), &right.into())
            }}
        }
        BinOp::Le(_) => {
            let left_expr = replace_expressions(*left, constants);
            let right_expr = replace_expressions(*right, constants);
            syn::parse_quote! {{
                let left = #left_expr;
                let right = #right_expr;
                context.le(&left.into(), &right.into())
            }}
        }
        // Arithmetic operators
        BinOp::Add(_) => {
            let left_expr = replace_expressions(*left, constants);
            let right_expr = replace_expressions(*right, constants);
            syn::parse_quote! {{
                let left = &#left_expr;
                let right = &#right_expr;
                context.add(left.into(), right.into())
            }}
        }
        BinOp::AddAssign(_) => {
            syn::parse_quote! {
                context.add(&#left, &#right)
            }
        }
        BinOp::Sub(_) => {
            let left_expr = replace_expressions(*left, constants);
            let right_expr = replace_expressions(*right, constants);
            syn::parse_quote! {{
                let left = #left_expr;
                let right = #right_expr;
                context.sub(&left.into(), &right.into())
            }}
        }
        BinOp::SubAssign(_) => {
            syn::parse_quote! {
                context.sub(&#left, &#right)
            }
        }
        BinOp::Mul(_) => {
            let left_expr = replace_expressions(*left, constants);
            let right_expr = replace_expressions(*right, constants);
            syn::parse_quote! {{
                let left = &#left_expr;
                let right = &#right_expr;
                context.mul(left.into(), right.into())
            }}
        }
        BinOp::MulAssign(_) => {
            syn::parse_quote! {
                context.mul(&#left, &#right)
            }
        }
        BinOp::Div(_) => {
            let left_expr = replace_expressions(*left, constants);
            let right_expr = replace_expressions(*right, constants);
            syn::parse_quote! {{
                let left = #left_expr;
                let right = #right_expr;
                context.div(&left.into(), &right.into())
            }}
        }
        BinOp::DivAssign(_) => {
            syn::parse_quote! {
                context.div(&#left, &#right)
            }
        }
        BinOp::Rem(_) => {
            let left_expr = replace_expressions(*left, constants);
            let right_expr = replace_expressions(*right, constants);
            syn::parse_quote! {{
                let left = #left_expr;
                let right = #right_expr;
                context.rem(&left.into(), &right.into())
            }}
        }
        BinOp::RemAssign(_) => {
            syn::parse_quote! {
                context.rem(&#left, &#right)
            }
        }
        // Logical operators
        BinOp::And(_) => {
            let left_expr = replace_expressions(*left, constants);
            let right_expr = replace_expressions(*right, constants);
            syn::parse_quote! {{
                let left = #left_expr;
                let right = #right_expr;
                context.land(&left, &right)
            }}
        }
        BinOp::Or(_) => {
            let left_expr = replace_expressions(*left, constants);
            let right_expr = replace_expressions(*right, constants);
            syn::parse_quote! {{
                let left = #left_expr;
                let right = #right_expr;
                context.lor(&left, &right)
            }}
        }
        // Bitwise operators
        BinOp::BitAnd(_) => {
            let left_expr = replace_expressions(*left, constants);
            let right_expr = replace_expressions(*right, constants);
            syn::parse_quote! {{
                let left = #left_expr;
                let right = #right_expr;
                context.and(&left.into(), &right.into())
            }}
        }
        BinOp::BitAndAssign(_) => {
            syn::parse_quote! {
                context.and(&#left, &#right)
            }
        }
        BinOp::BitOr(_) => {
            let left_expr = replace_expressions(*left, constants);
            let right_expr = replace_expressions(*right, constants);
            syn::parse_quote! {{
                let left = #left_expr;
                let right = #right_expr;
                context.or(&left.into(), &right.into())
            }}
        }
        BinOp::BitOrAssign(_) => {
            syn::parse_quote! {
                context.or(&#left, &#right)
            }
        }
        BinOp::BitXor(_) => {
            let left_expr = replace_expressions(*left, constants);
            let right_expr = replace_expressions(*right, constants);
            syn::parse_quote! {{
                let left = #left_expr;
                let right = #right_expr;
                context.xor(&left.into(), &right.into())
            }}
        }
        BinOp::BitXorAssign(_) => {
            syn::parse_quote! {
                context.xor(&#left, &#right)
            }
        }
        _ => panic!("Unsupported binary operator"),
    }
}

fn handle_unary_expression(expr: ExprUnary, constants: &mut Vec<TokenStream>) -> Expr {
    let ExprUnary { op, expr, .. } = expr;

    match op {
        syn::UnOp::Not(_) => {
            let single_expr = replace_expressions(*expr, constants);
            syn::parse_quote! {{
                let single = #single_expr;
                context.not(&single.into())
            }}
        }
        _ => panic!("Unsupported unary operator"),
    }
}

fn handle_if_expression(expr_if: ExprIf, constants: &mut Vec<TokenStream>) -> Expr {
    let ExprIf {
        cond,
        then_branch,
        else_branch,
        ..
    } = expr_if;

    // Check if `cond` is an `if let` with a range pattern
    let cond_expr = match *cond {
        Expr::Let(ExprLet { pat, expr, .. }) => handle_if_let_pattern(pat, expr, constants),
        ref _other => replace_expressions(*cond, constants),
    };

    let then_block = modify_body(then_branch, constants);

    // Check if an `else` branch exists, as it's required.
    let else_expr = if let Some((_, else_expr)) = else_branch {
        replace_expressions(*else_expr, constants)
    } else {
        panic!("else branch is required for if expressions");
    };

    // Generate code for conditional execution
    syn::parse_quote! {{
        let cond = #cond_expr;
        let if_true = #then_block;
        let if_false = #else_expr;
        context.mux(&cond.into(), &if_true, &if_false)
    }}
}

fn handle_if_let_pattern(pat: Box<Pat>, expr: Box<Expr>, constants: &mut Vec<TokenStream>) -> Expr {
    match &*pat {
        // Handle inclusive range pattern (e.g., 1..=5)
        syn::Pat::Range(syn::PatRange {
            start: Some(start),
            end: Some(end),
            limits: syn::RangeLimits::Closed(_),
            ..
        }) => {
            let start_expr = replace_expressions(*start.clone(), constants);
            let end_expr = replace_expressions(*end.clone(), constants);
            let input_expr = replace_expressions(*expr, constants);

            syn::parse_quote! {{
                let lhs = &context.ge(&#input_expr.into(), &#start_expr.into()).into();
                let rhs = &context.le(&#input_expr.into(), &#end_expr.into()).into();
                context.and(lhs, rhs)
            }}
        }
        // Handle exclusive range pattern (e.g., 1..10)
        syn::Pat::Range(syn::PatRange {
            start: Some(start),
            end: Some(end),
            limits: syn::RangeLimits::HalfOpen(_),
            ..
        }) => {
            let start_expr = replace_expressions(*start.clone(), constants);
            let end_expr = replace_expressions(*end.clone(), constants);
            let input_expr = replace_expressions(*expr, constants);

            syn::parse_quote! {{
                let lhs = &context.ge(&#input_expr.into(), &#start_expr.into()).into();
                let rhs = &context.lt(&#input_expr.into(), &#end_expr.into()).into();
                context.and(lhs, rhs)
            }}
        }
        // Handle single literal pattern, e.g., `if let 5 = n`
        syn::Pat::Lit(lit) => {
            let lit_expr = replace_expressions(Expr::Lit(lit.clone()), constants);
            let input_expr = replace_expressions(*expr, constants);

            syn::parse_quote! {
                context.eq(&#input_expr.into(), &#lit_expr.into())
            }
        }
        _ => panic!("Unsupported pattern in if let: expected a range or literal pattern."),
    }
}

fn handle_match_expression(expr_match: ExprMatch, constants: &mut Vec<TokenStream>) -> Expr {
    let ExprMatch { expr, arms, .. } = expr_match;
    let match_expr = replace_expressions(*expr, constants);

    // Define an input variable to use in range proof processing
    let input = syn::Ident::new("input", proc_macro2::Span::call_site());
    let input_binding = quote! { let #input = #match_expr; };

    // Process each arm, building up the conditional chain
    let arm_exprs = arms
        .into_iter()
        .rev()
        .fold(None as Option<Expr>, |acc, arm| {
            let pat = arm.pat;
            let body_expr = replace_expressions(*arm.body, constants);

            // Create conditional expression for each arm, handling ranges
            let cond_expr = match &pat {
                // Handle inclusive range pattern (start..=end)
                syn::Pat::Range(syn::PatRange {
                    start: Some(start),
                    end: Some(end),
                    limits: syn::RangeLimits::Closed(_),
                    ..
                }) => {
                    let start = replace_expressions(*start.clone(), constants);
                    let end = replace_expressions(*end.clone(), constants);
                    quote! {
                        let lhs = &context.ge(&#input.into(), &#start.into()).into();
                        let rhs = &context.le(&#input.into(), &#end.into()).into();
                        context.and(
                            lhs,
                            rhs
                        )
                    }
                }
                // Handle exclusive range pattern (start..end)
                syn::Pat::Range(syn::PatRange {
                    start: Some(start),
                    end: Some(end),
                    limits: syn::RangeLimits::HalfOpen(_),
                    ..
                }) => {
                    let start = replace_expressions(*start.clone(), constants);
                    let end = replace_expressions(*end.clone(), constants);
                    quote! {
                        let lhs = &context.ge(&#input.into(), &#start.into()).into();
                        let rhs = &context.lt(&#input.into(), &#end.into()).into();
                        context.and(
                            lhs,
                            rhs
                        )
                    }
                }
                // Handle single value pattern (e.g., `5`)
                syn::Pat::Lit(lit) => {
                    let lit_expr = replace_expressions(syn::Expr::Lit(lit.clone()), constants);
                    quote! {
                        context.eq(&#input.into(), &#lit_expr.into())
                    }
                }

                syn::Pat::Ident(pat) => {
                    // Create conditional expression for each arm
                    let cond_expr = replace_expressions(
                        syn::parse_quote! { #match_expr == #pat },
                        constants,
                    );

                    syn::parse_quote! {{
                        { #cond_expr }
                    }}
                }
                // Handle the wildcard pattern `_` as default/fallback case
                syn::Pat::Wild(_) => quote! { true },
                other => panic!("{:?}: Unsupported pattern in match arm", other),
            };

            // Chain the condition with the body, selecting based on condition
            Some(if let Some(else_expr) = acc {
                syn::parse_quote! {{
                    let if_true = { #body_expr };
                    let if_false = { #else_expr };
                    let cond = { #cond_expr };
                    context.mux(&cond.into(), &if_true, &if_false)
                }}
            } else {
                syn::parse_quote! {{
                    { #body_expr }
                }}
            })
        });

    match arm_exprs {
        Some(result) => syn::parse_quote! {{
            #input_binding // Bind `input` at the beginning
            #result        // Process the chained expressions
        }},
        None => panic!("Match expression requires at least one arm"),
    }
}