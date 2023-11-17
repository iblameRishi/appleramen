"""Welcome to Reflex!."""

from appleramen.state import State
import reflex as rx

color = "rgb(107,99,246)"


def index() -> rx.component:

    return rx.vstack(
        rx.upload(
            rx.vstack(
                rx.button(
                    "+",
                    color=color,
                    bg="white",
                    border=f"1px solid {color}",
                ),
                rx.text(
                    "Drag and drop files here or click to select files"
                ),
            ),
            border=f"1px dashed {color} ",
            border_radius="0.5em",
            padding="5em",

        ),
        rx.hstack(rx.foreach(rx.selected_files, rx.text)),
        rx.hstack(
            rx.button(
                rx.box(
                    "Upload Image ",
                    rx.span(
                        "→", class_name="inline-block translate-x-0 group-hover:translate-x-1 transition-transform ease-in-out duration-200"),
                    class_name="group text-slate-700 hover:text-sky-600 transition ease-in-out duration-200"
                ),
                color=color,
                bg="white",
                border=f"1px solid {color}",
                on_click=lambda: State.handle_upload(
                    rx.upload_files()
                ),
            ),
            rx.button(
                "Clear",
                on_click=rx.clear_selected_files,
            ),
            padding="2em",
        ),
        rx.box(
            rx.text(f'Prediction: {State.pred}')
        ),
        class_name="flex items-center justify-center h-screen bg-[#FCEED2]",
    )


app = rx.App()
app.add_page(index)
app.compile()
