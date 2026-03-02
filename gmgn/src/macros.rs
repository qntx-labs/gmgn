//! Internal macros for reducing boilerplate in wrapper implementations.

/// Delegate common [`Env`](crate::env::Env) methods to an inner environment field.
///
/// Place this macro **inside** an `impl Env for Wrapper` block to generate
/// default delegations. Only methods you do NOT manually implement should
/// be included.
///
/// # Usage
///
/// ```rust,ignore
/// impl<E: Env> Env for MyWrapper<E> {
///     type Obs = E::Obs;
///     type Act = E::Act;
///     type ObsSpace = E::ObsSpace;
///     type ActSpace = E::ActSpace;
///
///     fn step(&mut self, action: &Self::Act) -> Result<StepResult<Self::Obs>> { /* custom */ }
///     fn reset(&mut self, seed: Option<u64>) -> Result<ResetResult<Self::Obs>> { /* custom */ }
///
///     // Delegate the remaining five methods:
///     delegate_env!(env);
/// }
/// ```
///
/// For wrappers that override `observation_space` or `action_space`, list
/// only the methods to delegate:
///
/// ```rust,ignore
/// delegate_env!(env, render, close, render_mode, observation_space);
/// // action_space is NOT delegated — implemented manually above.
/// ```
macro_rules! delegate_env {
    // Default: delegate all five "boring" methods.
    ($field:ident) => {
        delegate_env!($field, render, close, render_mode, observation_space, action_space);
    };

    // Selective: delegate only the listed methods.
    ($field:ident, $($method:ident),+ $(,)?) => {
        $(delegate_env!(@one $field, $method);)+
    };

    (@one $field:ident, render) => {
        fn render(&mut self) -> $crate::error::Result<$crate::env::RenderFrame> {
            self.$field.render()
        }
    };
    (@one $field:ident, close) => {
        fn close(&mut self) { self.$field.close(); }
    };
    (@one $field:ident, render_mode) => {
        fn render_mode(&self) -> &$crate::env::RenderMode { self.$field.render_mode() }
    };
    (@one $field:ident, observation_space) => {
        fn observation_space(&self) -> &Self::ObsSpace { self.$field.observation_space() }
    };
    (@one $field:ident, action_space) => {
        fn action_space(&self) -> &Self::ActSpace { self.$field.action_space() }
    };
    // Note: `step` delegation is intentionally omitted. Wrappers almost always
    // override `step` with custom logic, so a delegation arm would be misleading.
    (@one $field:ident, reset) => {
        fn reset(
            &mut self,
            seed: Option<u64>,
        ) -> $crate::error::Result<$crate::env::ResetResult<Self::Obs>> {
            self.$field.reset(seed)
        }
    };
}

pub(crate) use delegate_env;
