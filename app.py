import customtkinter as ctk
from datetime import datetime
from rag_pipeline import RAGPipeline

PDF_FOLDER = r"C:\Users\salsabil.guizani\OneDrive - CEGID\Desktop\Salsabil AI-POD\pdfs"
INDEX_PATH = "faiss_index.idx"
pipeline = RAGPipeline(pdf_folder=PDF_FOLDER, index_path=INDEX_PATH)
if not pipeline.prepare_pipeline():
    raise Exception("Error loading PDF documents.")

ctk.set_appearance_mode("light")  
ctk.set_default_color_theme("blue")  

class ChatApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("RAG Assistant")
        self.geometry("750x800")
        self.configure(padx=20, pady=20)
        self.resizable(False, False)

        self.title_label = ctk.CTkLabel(
            self,
            text="RAG Assistant",
            font=ctk.CTkFont("Arial", 28, "bold"),
            text_color="#1f6aa5"
        )
        self.title_label.pack(pady=(0, 20))

        self.chat_box = ctk.CTkTextbox(
            self,
            width=700,
            height=580,
            font=("Arial", 14),
            wrap="word",
            corner_radius=10,
            border_width=2,
            border_color="#dddddd"
        )
        self.chat_box.pack(pady=(0, 10))
        self.chat_box.configure(state="disabled")

        self.entry_frame = ctk.CTkFrame(self, fg_color="#f2f2f2")
        self.entry_frame.pack(fill="x", pady=10, padx=5)

        self.user_input = ctk.CTkEntry(
            self.entry_frame,
            placeholder_text="Ask your question here...",
            width=540,
            height=40
        )
        self.user_input.pack(side="left", padx=10, pady=10)

        self.send_button = ctk.CTkButton(
            self.entry_frame,
            text="Send",
            command=self.send_message,
            width=100,
            height=40
        )
        self.send_button.pack(side="right", padx=10)

        self.bind("<Return>", lambda event: self.send_message())

    def send_message(self):
        user_message = self.user_input.get().strip()
        if not user_message:
            return

        self.display_message("👤 You", user_message, is_user=True)

        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", "\n🤖 Assistant is thinking...\n", "bot")
        self.chat_box.configure(state="disabled")
        self.chat_box.see("end")
        self.update_idletasks()

        bot_response = self.get_bot_response(user_message)
        self.display_message("🤖 Assistant", bot_response, is_user=False)

        self.user_input.delete(0, 'end')

    def display_message(self, sender, message, is_user=False):
        timestamp = datetime.now().strftime("%H:%M")
        formatted = f"\n[{timestamp}] {sender}:\n{message}\n"

        self.chat_box.configure(state="normal")

        if is_user:
            self.chat_box.insert("end", formatted, "user")
        else:
            self.chat_box.insert("end", formatted, "bot")

        self.chat_box.tag_config("user", foreground="#000000", background="#e0e0e0", spacing3=5)
        self.chat_box.tag_config("bot", foreground="#ffffff", background="#1f6aa5", spacing3=5)

        self.chat_box.configure(state="disabled")
        self.chat_box.see("end")

    def get_bot_response(self, message):
        try:
            return pipeline.search_and_respond(message)
        except Exception as e:
            return f"An error occurred: {e}"

if __name__ == "__main__":
    app = ChatApp()
    app.mainloop()
