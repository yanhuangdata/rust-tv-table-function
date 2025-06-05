
#[allow(dead_code)]
mod zngur_types {
    pub struct ZngurCppOpaqueBorrowedObject(());

    #[repr(C)]
    pub struct ZngurCppOpaqueOwnedObject {
        data: *mut u8,
        destructor: extern "C" fn(*mut u8),
    }

    impl ZngurCppOpaqueOwnedObject {
        pub unsafe fn new(
            data: *mut u8,
            destructor: extern "C" fn(*mut u8),            
        ) -> Self {
            Self { data, destructor }
        }

        pub fn ptr(&self) -> *mut u8 {
            self.data
        }
    }

    impl Drop for ZngurCppOpaqueOwnedObject {
        fn drop(&mut self) {
            (self.destructor)(self.data)
        }
    }
}

#[allow(unused_imports)]
pub use zngur_types::ZngurCppOpaqueOwnedObject;
#[allow(unused_imports)]
pub use zngur_types::ZngurCppOpaqueBorrowedObject;
thread_local! {
            pub static PANIC_PAYLOAD: ::std::cell::Cell<Option<()>> = ::std::cell::Cell::new(None);
        }
        #[allow(non_snake_case)]
        #[unsafe(no_mangle)]
        pub fn __zngur_detect_panic() -> u8 {
            PANIC_PAYLOAD.with(|p| {
                let pp = p.take();
                let r = if pp.is_some() { 1 } else { 0 };
                p.set(pp);
                r
            })
        }

        #[allow(non_snake_case)]
        #[unsafe(no_mangle)]
        pub fn __zngur_take_panic() {
            PANIC_PAYLOAD.with(|p| {
                p.take();
            })
        }
        
const _: [(); 24] = [(); ::std::mem::size_of::<::std::string::String>()];
const _: [(); 8] = [(); ::std::mem::align_of::<::std::string::String>()];

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur__std_string_String_drop_in_place_s8s12s19e26(v: *mut u8) { unsafe {
    ::std::ptr::drop_in_place(v as *mut ::std::string::String);
} }

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur___std_string_String_as_str___x8s9s13s20n27m34y35(i0: *mut u8, o: *mut u8) { unsafe {
let e = ::std::panic::catch_unwind(|| {
    ::std::ptr::write(o as *mut &str, <::std::string::String>::as_str::<>(::std::ptr::read(i0 as *mut &::std::string::String), ));
});
if let Err(_) = e { PANIC_PAYLOAD.with(|p| p.set(Some(()))) }
 } }

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur___std_string_String_clone___x8s9s13s20n27m33y34(i0: *mut u8, o: *mut u8) { unsafe {
let e = ::std::panic::catch_unwind(|| {
    ::std::ptr::write(o as *mut ::std::string::String, <::std::string::String>::clone::<>(::std::ptr::read(i0 as *mut &::std::string::String), ));
});
if let Err(_) = e { PANIC_PAYLOAD.with(|p| p.set(Some(()))) }
 } }
const _: [(); 24] = [(); ::std::mem::size_of::<::std::vec::Vec::<::rust_tvtf_api::FunctionRegistry>>()];
const _: [(); 8] = [(); ::std::mem::align_of::<::std::vec::Vec::<::rust_tvtf_api::FunctionRegistry>>()];

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur__std_vec_Vec__rust_tvtf_api_FunctionRegistry__drop_in_place_s8s12s16m20s21s35y52e53(v: *mut u8) { unsafe {
    ::std::ptr::drop_in_place(v as *mut ::std::vec::Vec::<::rust_tvtf_api::FunctionRegistry>);
} }

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur___std_vec_Vec__rust_tvtf_api_FunctionRegistry__len___x8s9s13s17m21s22s36y53n54m58y59(i0: *mut u8, o: *mut u8) { unsafe {
let e = ::std::panic::catch_unwind(|| {
    ::std::ptr::write(o as *mut usize, <::std::vec::Vec::<::rust_tvtf_api::FunctionRegistry>>::len::<>(::std::ptr::read(i0 as *mut &::std::vec::Vec::<::rust_tvtf_api::FunctionRegistry>), ));
});
if let Err(_) = e { PANIC_PAYLOAD.with(|p| p.set(Some(()))) }
 } }

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur___std_vec_Vec__rust_tvtf_api_FunctionRegistry__new___x8s9s13s17m21s22s36y53n54m58y59(o: *mut u8) { unsafe {
let e = ::std::panic::catch_unwind(|| {
    ::std::ptr::write(o as *mut ::std::vec::Vec::<::rust_tvtf_api::FunctionRegistry>, <::std::vec::Vec::<::rust_tvtf_api::FunctionRegistry>>::new::<>());
});
if let Err(_) = e { PANIC_PAYLOAD.with(|p| p.set(Some(()))) }
 } }

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur____rust_tvtf_api_FunctionRegistry__get___x8j9s10s24k41n42m46y47_deref___zngur___std_vec_Vec__rust_tvtf_api_FunctionRegistry__r8s9s13s17m21s22s36y53(i0: *mut u8, i1: *mut u8, o: *mut u8) { unsafe {
let e = ::std::panic::catch_unwind(|| {
    ::std::ptr::write(o as *mut ::std::option::Option::<&::rust_tvtf_api::FunctionRegistry>, <[::rust_tvtf_api::FunctionRegistry]>::get::<>(&::std::ptr::read(i0 as *mut &::std::vec::Vec::<::rust_tvtf_api::FunctionRegistry>), ::std::ptr::read(i1 as *mut usize), ));
});
if let Err(_) = e { PANIC_PAYLOAD.with(|p| p.set(Some(()))) }
 } }
const _: [(); 16] = [(); ::std::mem::size_of::<Box<dyn ::rust_tvtf_api::TableFunction>>()];
const _: [(); 8] = [(); ::std::mem::align_of::<Box<dyn ::rust_tvtf_api::TableFunction>>()];

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur_Box_dyn_rust_tvtf_api_TableFunction__drop_in_place_x11s15s29y43e44(v: *mut u8) { unsafe {
    ::std::ptr::drop_in_place(v as *mut Box<dyn ::rust_tvtf_api::TableFunction>);
} }
const _: [(); 8] = [(); ::std::mem::size_of::<::std::option::Option::<&::rust_tvtf_api::FunctionRegistry>>()];
const _: [(); 8] = [(); ::std::mem::align_of::<::std::option::Option::<&::rust_tvtf_api::FunctionRegistry>>()];

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur__std_option_Option___rust_tvtf_api_FunctionRegistry__drop_in_place_s8s12s19m26r27s28s42y59e60(v: *mut u8) { unsafe {
    ::std::ptr::drop_in_place(v as *mut ::std::option::Option::<&::rust_tvtf_api::FunctionRegistry>);
} }

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur___std_option_Option___rust_tvtf_api_FunctionRegistry__is_none___x8s9s13s20m27r28s29s43y60n61m69y70(i0: *mut u8, o: *mut u8) { unsafe {
let e = ::std::panic::catch_unwind(|| {
    ::std::ptr::write(o as *mut bool, <::std::option::Option::<&::rust_tvtf_api::FunctionRegistry>>::is_none::<>(::std::ptr::read(i0 as *mut &::std::option::Option::<&::rust_tvtf_api::FunctionRegistry>), ));
});
if let Err(_) = e { PANIC_PAYLOAD.with(|p| p.set(Some(()))) }
 } }

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur___std_option_Option___rust_tvtf_api_FunctionRegistry__is_some___x8s9s13s20m27r28s29s43y60n61m69y70(i0: *mut u8, o: *mut u8) { unsafe {
let e = ::std::panic::catch_unwind(|| {
    ::std::ptr::write(o as *mut bool, <::std::option::Option::<&::rust_tvtf_api::FunctionRegistry>>::is_some::<>(::std::ptr::read(i0 as *mut &::std::option::Option::<&::rust_tvtf_api::FunctionRegistry>), ));
});
if let Err(_) = e { PANIC_PAYLOAD.with(|p| p.set(Some(()))) }
 } }

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur___std_option_Option___rust_tvtf_api_FunctionRegistry__unwrap___x8s9s13s20m27r28s29s43y60n61m68y69(i0: *mut u8, o: *mut u8) { unsafe {
let e = ::std::panic::catch_unwind(|| {
    ::std::ptr::write(o as *mut &::rust_tvtf_api::FunctionRegistry, <::std::option::Option::<&::rust_tvtf_api::FunctionRegistry>>::unwrap::<>(::std::ptr::read(i0 as *mut ::std::option::Option::<&::rust_tvtf_api::FunctionRegistry>), ));
});
if let Err(_) = e { PANIC_PAYLOAD.with(|p| p.set(Some(()))) }
 } }
const _: [(); 8] = [(); ::std::mem::size_of::<::anyhow::Error>()];
const _: [(); 8] = [(); ::std::mem::align_of::<::anyhow::Error>()];

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur__anyhow_Error_drop_in_place_s8s15e21(v: *mut u8) { unsafe {
    ::std::ptr::drop_in_place(v as *mut ::anyhow::Error);
} }

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur___anyhow_Error_to_string___x8s9s16n22m32y33(i0: *mut u8, o: *mut u8) { unsafe {
let e = ::std::panic::catch_unwind(|| {
    ::std::ptr::write(o as *mut ::std::string::String, <::anyhow::Error>::to_string::<>(::std::ptr::read(i0 as *mut &::anyhow::Error), ));
});
if let Err(_) = e { PANIC_PAYLOAD.with(|p| p.set(Some(()))) }
 } }
const _: [(); 16] = [(); ::std::mem::size_of::<::std::result::Result::<i64, ::anyhow::Error>>()];
const _: [(); 8] = [(); ::std::mem::align_of::<::std::result::Result::<i64, ::anyhow::Error>>()];

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur__std_result_Result_i64__anyhow_Error_Ok_s8s12s19m26c30s31s38n44(f_0: *mut u8, o: *mut u8) { unsafe {
    ::std::ptr::write(o as *mut _, ::std::result::Result::<i64, ::anyhow::Error>::Ok { 0: ::std::ptr::read(f_0 as *mut i64), }) } }

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur__std_result_Result_i64__anyhow_Error_Ok_s8s12s19m26c30s31s38n44_check(i: *mut u8, o: *mut u8) { unsafe {
    *o = matches!(&*(i as *mut &_), ::std::result::Result::<i64, ::anyhow::Error>::Ok { .. }) as u8;
} }
#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur__std_result_Result_i64__anyhow_Error__drop_in_place_s8s12s19m26c30s31s38y44e45(v: *mut u8) { unsafe {
    ::std::ptr::drop_in_place(v as *mut ::std::result::Result::<i64, ::anyhow::Error>);
} }

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur___std_result_Result_i64__anyhow_Error__is_ok___x8s9s13s20m27c31s32s39y45n46m52y53(i0: *mut u8, o: *mut u8) { unsafe {
let e = ::std::panic::catch_unwind(|| {
    ::std::ptr::write(o as *mut bool, <::std::result::Result::<i64, ::anyhow::Error>>::is_ok::<>(::std::ptr::read(i0 as *mut &::std::result::Result::<i64, ::anyhow::Error>), ));
});
if let Err(_) = e { PANIC_PAYLOAD.with(|p| p.set(Some(()))) }
 } }

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur___std_result_Result_i64__anyhow_Error__is_err___x8s9s13s20m27c31s32s39y45n46m53y54(i0: *mut u8, o: *mut u8) { unsafe {
let e = ::std::panic::catch_unwind(|| {
    ::std::ptr::write(o as *mut bool, <::std::result::Result::<i64, ::anyhow::Error>>::is_err::<>(::std::ptr::read(i0 as *mut &::std::result::Result::<i64, ::anyhow::Error>), ));
});
if let Err(_) = e { PANIC_PAYLOAD.with(|p| p.set(Some(()))) }
 } }

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur___std_result_Result_i64__anyhow_Error__unwrap___x8s9s13s20m27c31s32s39y45n46m53y54(i0: *mut u8, o: *mut u8) { unsafe {
let e = ::std::panic::catch_unwind(|| {
    ::std::ptr::write(o as *mut i64, <::std::result::Result::<i64, ::anyhow::Error>>::unwrap::<>(::std::ptr::read(i0 as *mut ::std::result::Result::<i64, ::anyhow::Error>), ));
});
if let Err(_) = e { PANIC_PAYLOAD.with(|p| p.set(Some(()))) }
 } }

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur___std_result_Result_i64__anyhow_Error__unwrap_err___x8s9s13s20m27c31s32s39y45n46m57y58(i0: *mut u8, o: *mut u8) { unsafe {
let e = ::std::panic::catch_unwind(|| {
    ::std::ptr::write(o as *mut ::anyhow::Error, <::std::result::Result::<i64, ::anyhow::Error>>::unwrap_err::<>(::std::ptr::read(i0 as *mut ::std::result::Result::<i64, ::anyhow::Error>), ));
});
if let Err(_) = e { PANIC_PAYLOAD.with(|p| p.set(Some(()))) }
 } }
const _: [(); 16] = [(); ::std::mem::size_of::<::std::result::Result::<Box<dyn ::rust_tvtf_api::TableFunction>, ::anyhow::Error>>()];
const _: [(); 8] = [(); ::std::mem::align_of::<::std::result::Result::<Box<dyn ::rust_tvtf_api::TableFunction>, ::anyhow::Error>>()];

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur__std_result_Result_Box_dyn_rust_tvtf_api_TableFunction___anyhow_Error_Ok_s8s12s19m26x30s34s48y62c63s64s71n77(f_0: *mut u8, o: *mut u8) { unsafe {
    ::std::ptr::write(o as *mut _, ::std::result::Result::<Box<dyn ::rust_tvtf_api::TableFunction>, ::anyhow::Error>::Ok { 0: ::std::ptr::read(f_0 as *mut Box<dyn ::rust_tvtf_api::TableFunction>), }) } }

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur__std_result_Result_Box_dyn_rust_tvtf_api_TableFunction___anyhow_Error_Ok_s8s12s19m26x30s34s48y62c63s64s71n77_check(i: *mut u8, o: *mut u8) { unsafe {
    *o = matches!(&*(i as *mut &_), ::std::result::Result::<Box<dyn ::rust_tvtf_api::TableFunction>, ::anyhow::Error>::Ok { .. }) as u8;
} }
#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur__std_result_Result_Box_dyn_rust_tvtf_api_TableFunction___anyhow_Error__drop_in_place_s8s12s19m26x30s34s48y62c63s64s71y77e78(v: *mut u8) { unsafe {
    ::std::ptr::drop_in_place(v as *mut ::std::result::Result::<Box<dyn ::rust_tvtf_api::TableFunction>, ::anyhow::Error>);
} }

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur___std_result_Result_Box_dyn_rust_tvtf_api_TableFunction___anyhow_Error__is_ok___x8s9s13s20m27x31s35s49y63c64s65s72y78n79m85y86(i0: *mut u8, o: *mut u8) { unsafe {
let e = ::std::panic::catch_unwind(|| {
    ::std::ptr::write(o as *mut bool, <::std::result::Result::<Box<dyn ::rust_tvtf_api::TableFunction>, ::anyhow::Error>>::is_ok::<>(::std::ptr::read(i0 as *mut &::std::result::Result::<Box<dyn ::rust_tvtf_api::TableFunction>, ::anyhow::Error>), ));
});
if let Err(_) = e { PANIC_PAYLOAD.with(|p| p.set(Some(()))) }
 } }

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur___std_result_Result_Box_dyn_rust_tvtf_api_TableFunction___anyhow_Error__is_err___x8s9s13s20m27x31s35s49y63c64s65s72y78n79m86y87(i0: *mut u8, o: *mut u8) { unsafe {
let e = ::std::panic::catch_unwind(|| {
    ::std::ptr::write(o as *mut bool, <::std::result::Result::<Box<dyn ::rust_tvtf_api::TableFunction>, ::anyhow::Error>>::is_err::<>(::std::ptr::read(i0 as *mut &::std::result::Result::<Box<dyn ::rust_tvtf_api::TableFunction>, ::anyhow::Error>), ));
});
if let Err(_) = e { PANIC_PAYLOAD.with(|p| p.set(Some(()))) }
 } }

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur___std_result_Result_Box_dyn_rust_tvtf_api_TableFunction___anyhow_Error__unwrap___x8s9s13s20m27x31s35s49y63c64s65s72y78n79m86y87(i0: *mut u8, o: *mut u8) { unsafe {
let e = ::std::panic::catch_unwind(|| {
    ::std::ptr::write(o as *mut Box<dyn ::rust_tvtf_api::TableFunction>, <::std::result::Result::<Box<dyn ::rust_tvtf_api::TableFunction>, ::anyhow::Error>>::unwrap::<>(::std::ptr::read(i0 as *mut ::std::result::Result::<Box<dyn ::rust_tvtf_api::TableFunction>, ::anyhow::Error>), ));
});
if let Err(_) = e { PANIC_PAYLOAD.with(|p| p.set(Some(()))) }
 } }

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur___std_result_Result_Box_dyn_rust_tvtf_api_TableFunction___anyhow_Error__unwrap_err_unchecked___x8s9s13s20m27x31s35s49y63c64s65s72y78n79m100y101(i0: *mut u8, o: *mut u8) { unsafe {
let e = ::std::panic::catch_unwind(|| {
    ::std::ptr::write(o as *mut ::anyhow::Error, <::std::result::Result::<Box<dyn ::rust_tvtf_api::TableFunction>, ::anyhow::Error>>::unwrap_err_unchecked::<>(::std::ptr::read(i0 as *mut ::std::result::Result::<Box<dyn ::rust_tvtf_api::TableFunction>, ::anyhow::Error>), ));
});
if let Err(_) = e { PANIC_PAYLOAD.with(|p| p.set(Some(()))) }
 } }

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur__str_as_ptr___x8n12m19y20(i0: *mut u8, o: *mut u8) { unsafe {
let e = ::std::panic::catch_unwind(|| {
    ::std::ptr::write(o as *mut *const u8, <str>::as_ptr::<>(::std::ptr::read(i0 as *mut &str), ));
});
if let Err(_) = e { PANIC_PAYLOAD.with(|p| p.set(Some(()))) }
 } }

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur__str_len___x8n12m16y17(i0: *mut u8, o: *mut u8) { unsafe {
let e = ::std::panic::catch_unwind(|| {
    ::std::ptr::write(o as *mut usize, <str>::len::<>(::std::ptr::read(i0 as *mut &str), ));
});
if let Err(_) = e { PANIC_PAYLOAD.with(|p| p.set(Some(()))) }
 } }

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur__str_to_owned___x8n12m21y22(i0: *mut u8, o: *mut u8) { unsafe {
let e = ::std::panic::catch_unwind(|| {
    ::std::ptr::write(o as *mut ::std::string::String, <str>::to_owned::<>(::std::ptr::read(i0 as *mut &str), ));
});
if let Err(_) = e { PANIC_PAYLOAD.with(|p| p.set(Some(()))) }
 } }
const _: [(); 1] = [(); ::std::mem::size_of::<bool>()];
const _: [(); 1] = [(); ::std::mem::align_of::<bool>()];
const _: () = {
                const fn static_assert_is_copy<T: Copy>() {}
                static_assert_is_copy::<bool>();
            };
const _: [(); 32] = [(); ::std::mem::size_of::<::rust_tvtf_api::FunctionRegistry>()];
const _: [(); 8] = [(); ::std::mem::align_of::<::rust_tvtf_api::FunctionRegistry>()];

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur__rust_tvtf_api_FunctionRegistry_drop_in_place_s8s22e39(v: *mut u8) { unsafe {
    ::std::ptr::drop_in_place(v as *mut ::rust_tvtf_api::FunctionRegistry);
} }

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur___rust_tvtf_api_FunctionRegistry_name___x8s9s23n40m45y46(i0: *mut u8, o: *mut u8) { unsafe {
let e = ::std::panic::catch_unwind(|| {
    ::std::ptr::write(o as *mut &str, <::rust_tvtf_api::FunctionRegistry>::name::<>(::std::ptr::read(i0 as *mut &::rust_tvtf_api::FunctionRegistry), ));
});
if let Err(_) = e { PANIC_PAYLOAD.with(|p| p.set(Some(()))) }
 } }
const _: [(); 0] = [(); ::std::mem::size_of::<()>()];
const _: [(); 1] = [(); ::std::mem::align_of::<()>()];
const _: () = {
                const fn static_assert_is_copy<T: Copy>() {}
                static_assert_is_copy::<()>();
            };

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur__rust_tvtf_api_create_raw_s8s22(i0: *mut u8, i1: *mut u8, i2: *mut u8, o: *mut u8) { unsafe {
let e = ::std::panic::catch_unwind(|| {
    ::std::ptr::write(o as *mut ::std::result::Result::<Box<dyn ::rust_tvtf_api::TableFunction>, ::anyhow::Error>, ::rust_tvtf_api::create_raw(::std::ptr::read(i0 as *mut &::rust_tvtf_api::FunctionRegistry), ::std::ptr::read(i1 as *mut *const i8), ::std::ptr::read(i2 as *mut *const i8), ));
});
if let Err(_) = e { PANIC_PAYLOAD.with(|p| p.set(Some(()))) }
 } }

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur__rust_tvtf_api_process_raw_s8s22(i0: *mut u8, i1: *mut u8, o: *mut u8) { unsafe {
let e = ::std::panic::catch_unwind(|| {
    ::std::ptr::write(o as *mut ::std::result::Result::<i64, ::anyhow::Error>, ::rust_tvtf_api::process_raw(::std::ptr::read(i0 as *mut &mut Box<dyn ::rust_tvtf_api::TableFunction>), ::std::ptr::read(i1 as *mut i64), ));
});
if let Err(_) = e { PANIC_PAYLOAD.with(|p| p.set(Some(()))) }
 } }

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur__rust_tvtf_api_finalize_raw_s8s22(i0: *mut u8, o: *mut u8) { unsafe {
let e = ::std::panic::catch_unwind(|| {
    ::std::ptr::write(o as *mut ::std::result::Result::<i64, ::anyhow::Error>, ::rust_tvtf_api::finalize_raw(::std::ptr::read(i0 as *mut &mut Box<dyn ::rust_tvtf_api::TableFunction>), ));
});
if let Err(_) = e { PANIC_PAYLOAD.with(|p| p.set(Some(()))) }
 } }

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn __zngur_crate_get_function_registries_s13(o: *mut u8) { unsafe {
let e = ::std::panic::catch_unwind(|| {
    ::std::ptr::write(o as *mut ::std::vec::Vec::<::rust_tvtf_api::FunctionRegistry>, crate::get_function_registries());
});
if let Err(_) = e { PANIC_PAYLOAD.with(|p| p.set(Some(()))) }
 } }
