"""
Streamlit 用エントリポイント（クラウド公開向け）。

ローカル確認:
  pip install -r requirements.txt
  streamlit run streamlit_app.py

GitHub 連携: Streamlit Community Cloud
  https://streamlit.io/cloud
  → リポジトリを接続し、Main file に streamlit_app.py を指定。
"""
import matplotlib
from functools import partial

matplotlib.use("Agg")

import streamlit as st

from epsilon_delta import EpsilonDeltaVisualizer


def _sync_num_from_slider(slider_key: str, num_key: str) -> None:
    st.session_state[num_key] = st.session_state[slider_key]


def _sync_slider_from_num(slider_key: str, num_key: str) -> None:
    st.session_state[slider_key] = st.session_state[num_key]


@st.cache_resource
def get_visualizer() -> EpsilonDeltaVisualizer:
    return EpsilonDeltaVisualizer(streamlit_mode=True)


def _init_sidebar_state(viz: EpsilonDeltaVisualizer) -> None:
    if "epsilon_delta_ui_v2" in st.session_state:
        return
    st.session_state.epsilon_delta_ui_v2 = True
    st.session_state.sa = float(viz.initial_a)
    st.session_state.seps = float(viz.initial_epsilon)
    st.session_state.sdelta = float(viz.initial_delta)
    st.session_state.sb = float(viz.initial_b)
    st.session_state.sa_num = st.session_state.sa
    st.session_state.seps_num = st.session_state.seps
    st.session_state.sdelta_num = st.session_state.sdelta
    st.session_state.sb_num = st.session_state.sb
    st.session_state.func_expr_key = viz.initial_function_expr
    st.session_state.view_xlim = tuple(viz.initial_xlim)
    st.session_state.view_ylim = tuple(viz.initial_ylim)


def _apply_epsilon_preset(val: float) -> None:
    """ε プリセット（スライダー・数値入力を同時更新）"""
    eps_val = float(val)
    st.session_state.seps = eps_val
    st.session_state.seps_num = eps_val


def _apply_pending_streamlit_mutations(viz: EpsilonDeltaVisualizer) -> None:
    """ウィジェットより前に session_state を書き換え（Streamlit の制約対策）"""
    if st.session_state.pop("_ed_b_zero", False):
        st.session_state.sb = 0.0
        st.session_state.sb_num = 0.0
    if st.session_state.pop("_ed_full_reset", False):
        st.session_state.sa = float(viz.initial_a)
        st.session_state.seps = float(viz.initial_epsilon)
        st.session_state.sdelta = float(viz.initial_delta)
        st.session_state.sb = float(viz.initial_b)
        st.session_state.sa_num = st.session_state.sa
        st.session_state.seps_num = st.session_state.seps
        st.session_state.sdelta_num = st.session_state.sdelta
        st.session_state.sb_num = st.session_state.sb
        st.session_state.func_expr_key = viz.initial_function_expr
        st.session_state.view_xlim = tuple(viz.initial_xlim)
        st.session_state.view_ylim = tuple(viz.initial_ylim)


def _render_plot(viz: EpsilonDeltaVisualizer, plot_slot) -> None:
    expr = st.session_state.func_expr_key
    viz.a = float(st.session_state.sa)
    # スライダーと数値入力のずれを防ぎ、常に同期済みの値を使う
    viz.epsilon = float(st.session_state.get("seps_num", st.session_state.seps))
    viz.delta = float(st.session_state.get("sdelta_num", st.session_state.sdelta))
    viz.b = float(st.session_state.get("sb_num", st.session_state.sb))

    if "view_xlim" in st.session_state and "view_ylim" in st.session_state:
        viz.ax.set_xlim(st.session_state.view_xlim)
        viz.ax.set_ylim(st.session_state.view_ylim)

    if expr != viz.function_expr:
        viz.update_function(expr)
    else:
        viz.update(None)

    st.session_state.view_xlim = tuple(viz.ax.get_xlim())
    st.session_state.view_ylim = tuple(viz.ax.get_ylim())
    viz.fig.canvas.draw()
    with plot_slot:
        st.pyplot(viz.fig, use_container_width=True, dpi=72, clear_figure=False)


st.set_page_config(
    page_title="ε-δ論法",
    layout="wide",
    page_icon="📐",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": None,
    },
)

# 右上のメニュー（⋮）・Deploy 等を非表示。サイドバーは常に開いたままにする。
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer[data-testid="stFooter"] {visibility: hidden;}
        header[data-testid="stHeader"] {display: none;}
        div[data-testid="stToolbar"] {display: none;}
        div[data-testid="stDecoration"] {display: none;}
        div[data-testid="stAppDeployButton"] {display: none;}

        [data-testid="stSidebarCollapseButton"],
        [data-testid="stExpandSidebarButton"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"] {
            display: none !important;
            visibility: hidden !important;
            pointer-events: none !important;
            width: 0 !important;
            height: 0 !important;
            overflow: hidden !important;
            opacity: 0 !important;
        }

        section[data-testid="stSidebar"],
        [data-testid="stSidebar"] {
            transform: translateX(0) !important;
            visibility: visible !important;
            width: 32% !important;
            min-width: 22rem !important;
            max-width: 36rem !important;
        }
        section[data-testid="stSidebar"] > div {
            width: 100% !important;
        }
        [data-testid="stSidebarContent"] {
            width: 100% !important;
            opacity: 1 !important;
            visibility: visible !important;
        }
        [data-testid="stSidebar"][aria-expanded="false"] {
            transform: translateX(0) !important;
            margin-left: 0 !important;
        }

        /* 右メイン：自然な位置に配置・縦スクロール不可 */
        .stApp {
            overflow: hidden !important;
        }
        [data-testid="stAppViewContainer"] {
            overflow: hidden !important;
            height: 100vh !important;
        }
        [data-testid="stAppViewContainer"] > .main {
            overflow: hidden !important;
            height: 100vh !important;
        }
        div[data-testid="stMain"] {
            overflow: hidden !important;
            height: 100vh !important;
            max-height: 100vh !important;
        }
        section.main > div.block-container {
            padding-top: 0.5rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-bottom: 1rem !important;
            margin: 0 auto !important;
            max-width: 100% !important;
            overflow: hidden !important;
            box-sizing: border-box !important;
        }
        section.main h1 {
            text-align: left !important;
            margin: 0 0 0.5rem 0 !important;
            padding: 0 !important;
            font-size: 2.25rem !important;
            line-height: 1.2 !important;
        }
        div[data-testid="stMain"] div[data-testid="stPyplotGlobalUseContainerWidth"] {
            margin-top: 0 !important;
        }
        div[data-testid="stMain"] div[data-testid="stPyplotGlobalUseContainerWidth"] img,
        div[data-testid="stMain"] div[data-testid="stPyplotGlobalUseContainerWidth"] svg {
            display: block !important;
            margin: 0 !important;
            max-width: 100% !important;
            height: auto !important;
        }
        /* 古いグラフ画像が残らないよう、直近1枚だけ表示 */
        div[data-testid="stMain"] div[data-testid="stVerticalBlock"] > div[data-testid="element-container"]:has([data-testid="stPyplotGlobalUseContainerWidth"]) ~ div[data-testid="element-container"]:has([data-testid="stPyplotGlobalUseContainerWidth"]) {
            display: none !important;
        }
        /* サイドバーだけ縦スクロール可 */
        section[data-testid="stSidebar"] {
            overflow-y: auto !important;
            height: 100vh !important;
            max-height: 100vh !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "sidebar_lock_injected" not in st.session_state:
    st.components.v1.html(
        """
        <script>
        (function () {
            const doc = window.parent.document;
            const hideControls = () => {
                doc.querySelectorAll(
                    '[data-testid="stSidebarCollapseButton"],' +
                    '[data-testid="stExpandSidebarButton"],' +
                    '[data-testid="stSidebarCollapsedControl"],' +
                    '[data-testid="collapsedControl"]'
                ).forEach((el) => {
                    el.style.display = "none";
                    el.style.pointerEvents = "none";
                });
                const sidebar = doc.querySelector('[data-testid="stSidebar"]');
                if (sidebar) {
                    sidebar.setAttribute("aria-expanded", "true");
                    sidebar.style.transform = "translateX(0)";
                }
                Object.keys(window.parent.localStorage).forEach((key) => {
                    if (key.startsWith("stSidebarCollapsed")) {
                        window.parent.localStorage.setItem(key, "false");
                    }
                });
                const main = doc.querySelector('[data-testid="stMain"]');
                if (main) {
                    main.style.overflow = "hidden";
                    main.scrollTop = 0;
                }
                doc.querySelectorAll(
                    '[data-testid="stMain"] [data-testid="stPyplotGlobalUseContainerWidth"]'
                ).forEach((el, idx, arr) => {
                    if (idx < arr.length - 1) {
                        const container = el.closest('[data-testid="element-container"]');
                        if (container) container.style.display = "none";
                    }
                });
            };
            hideControls();
            window.setTimeout(hideControls, 100);
            window.setTimeout(hideControls, 500);
        })();
        </script>
        """,
        height=0,
        width=0,
    )
    st.session_state.sidebar_lock_injected = True

viz = get_visualizer()
_init_sidebar_state(viz)
_apply_pending_streamlit_mutations(viz)

st.title("ε-δ論法")
plot_slot = st.empty()

for _sk, _nk in (
    ("sa", "sa_num"),
    ("seps", "seps_num"),
    ("sdelta", "sdelta_num"),
    ("sb", "sb_num"),
):
    if _nk not in st.session_state and _sk in st.session_state:
        st.session_state[_nk] = st.session_state[_sk]

with st.sidebar:
    st.subheader("パラメータ")

    st.slider(
        "a（スライダー）",
        min_value=0.0,
        max_value=3.0,
        step=0.01,
        format="%.2f",
        key="sa",
        on_change=partial(_sync_num_from_slider, "sa", "sa_num"),
    )
    st.number_input(
        "a（数値入力）",
        min_value=0.0,
        max_value=3.0,
        step=0.01,
        format="%.2f",
        key="sa_num",
        on_change=partial(_sync_slider_from_num, "sa", "sa_num"),
    )

    st.slider(
        "ε（スライダー）",
        min_value=0.001,
        max_value=2.0,
        step=0.001,
        format="%.3f",
        key="seps",
        on_change=partial(_sync_num_from_slider, "seps", "seps_num"),
    )
    eps0, eps1, eps2 = st.columns(3)
    with eps0:
        st.button(
            "0.5",
            key="eps_preset_05",
            on_click=_apply_epsilon_preset,
            args=(0.5,),
        )
    with eps1:
        st.button(
            "0.02",
            key="eps_preset_002",
            on_click=_apply_epsilon_preset,
            args=(0.02,),
        )
    with eps2:
        st.button(
            "0.001",
            key="eps_preset_0001",
            on_click=_apply_epsilon_preset,
            args=(0.001,),
        )
    st.number_input(
        "ε（数値入力）",
        min_value=0.001,
        max_value=2.0,
        step=0.001,
        format="%.3f",
        key="seps_num",
        on_change=partial(_sync_slider_from_num, "seps", "seps_num"),
    )

    st.slider(
        "δ（スライダー）",
        min_value=0.0001,
        max_value=1.0,
        step=0.0001,
        format="%.4f",
        key="sdelta",
        on_change=partial(_sync_num_from_slider, "sdelta", "sdelta_num"),
    )
    st.number_input(
        "δ（数値入力）",
        min_value=0.0001,
        max_value=1.0,
        step=0.0001,
        format="%.4f",
        key="sdelta_num",
        on_change=partial(_sync_slider_from_num, "sdelta", "sdelta_num"),
    )

    st.slider(
        "b（スライダー）",
        min_value=-2.0,
        max_value=2.0,
        step=0.01,
        format="%.2f",
        key="sb",
        on_change=partial(_sync_num_from_slider, "sb", "sb_num"),
    )
    st.number_input(
        "b（数値入力）",
        min_value=-2.0,
        max_value=2.0,
        step=0.01,
        format="%.2f",
        key="sb_num",
        on_change=partial(_sync_slider_from_num, "sb", "sb_num"),
    )

    st.subheader("関数 f(x)")
    r1, r2 = st.columns(2)
    with r1:
        if st.button("x"):
            st.session_state.func_expr_key = "x"
            st.rerun()
        if st.button("x**2"):
            st.session_state.func_expr_key = "x**2"
            st.rerun()
    with r2:
        if st.button("sqrt(x)"):
            st.session_state.func_expr_key = "sqrt(x)"
            st.rerun()
        if st.button("±", help="現在の式の符号を反転"):
            viz.negate_function(None)
            st.session_state.func_expr_key = viz.function_expr
            st.rerun()

    st.text_input(
        "式（例: x**2, sqrt(x)）",
        key="func_expr_key",
    )

    st.subheader("表示")
    st.caption(
        "クラウド版は画像表示のためグラフ上のドラッグは使えません。"
        "矢印で見る範囲を移動できます。"
    )
    xlim = viz.ax.get_xlim()
    ylim = viz.ax.get_ylim()
    step_x = 0.12 * (xlim[1] - xlim[0])
    step_y = 0.12 * (ylim[1] - ylim[0])
    p0, p1, p2, p3 = st.columns(4)
    with p0:
        if st.button("◀", help="左へ"):
            viz.pan_by_data(step_x, 0.0)
            st.session_state.view_xlim = tuple(viz.ax.get_xlim())
            st.session_state.view_ylim = tuple(viz.ax.get_ylim())
    with p1:
        if st.button("▶", help="右へ"):
            viz.pan_by_data(-step_x, 0.0)
            st.session_state.view_xlim = tuple(viz.ax.get_xlim())
            st.session_state.view_ylim = tuple(viz.ax.get_ylim())
    with p2:
        if st.button("▲", help="上へ"):
            viz.pan_by_data(0.0, -step_y)
            st.session_state.view_xlim = tuple(viz.ax.get_xlim())
            st.session_state.view_ylim = tuple(viz.ax.get_ylim())
    with p3:
        if st.button("▼", help="下へ"):
            viz.pan_by_data(0.0, step_y)
            st.session_state.view_xlim = tuple(viz.ax.get_xlim())
            st.session_state.view_ylim = tuple(viz.ax.get_ylim())

    if st.button("b を 0 に"):
        st.session_state._ed_b_zero = True
        st.rerun()
    z1, z2, z3, z4 = st.columns(4)
    with z1:
        if st.button("拡大"):
            viz.zoom_in(None)
            st.session_state.view_xlim = tuple(viz.ax.get_xlim())
            st.session_state.view_ylim = tuple(viz.ax.get_ylim())
    with z2:
        if st.button("縮小"):
            viz.zoom_out(None)
            st.session_state.view_xlim = tuple(viz.ax.get_xlim())
            st.session_state.view_ylim = tuple(viz.ax.get_ylim())
    with z3:
        if st.button("10倍拡大"):
            viz.zoom_in_fast(None)
            st.session_state.view_xlim = tuple(viz.ax.get_xlim())
            st.session_state.view_ylim = tuple(viz.ax.get_ylim())
    with z4:
        if st.button("10倍縮小"):
            viz.zoom_out_fast(None)
            st.session_state.view_xlim = tuple(viz.ax.get_xlim())
            st.session_state.view_ylim = tuple(viz.ax.get_ylim())
    if st.button("リセット（初期値・表示範囲）"):
        st.session_state._ed_full_reset = True
        st.rerun()

_render_plot(viz, plot_slot)
