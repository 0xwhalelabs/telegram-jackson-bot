"""
카카오톡 이모티콘 추출기 (PC 버전)
Original Android app by Dark Tornado (GPL 3.0)
PC port: Python + tkinter

API Endpoints:
  - Search: https://e.kakao.com/api/v1/search?query={query}&page=0&size=20
  - Detail: https://e.kakao.com/api/v1/items/t/{titleUrl}
  - Images: thumbnailUrls from detail response
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import urllib.request
import urllib.parse
import json
import os
import io
import threading
from pathlib import Path

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ─── API helpers ───────────────────────────────────────────────

def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read()


def search_emoticons(query: str, page: int = 0, size: int = 20) -> list:
    encoded = urllib.parse.quote(query, safe="")
    url = f"https://e.kakao.com/api/v1/search?query={encoded}&page={page}&size={size}"
    data = fetch_json(url)
    return data.get("result", {}).get("content", [])


def get_emoticon_thumbnails(title_url: str) -> list:
    url = f"https://e.kakao.com/api/v1/items/t/{title_url}"
    data = fetch_json(url)
    return data.get("result", {}).get("thumbnailUrls", [])


# ─── GUI ───────────────────────────────────────────────────────

class KakaoEmoticonApp:
    BG = "#1e1e2e"
    FG = "#cdd6f4"
    ACCENT = "#f9e2af"
    BTN_BG = "#89b4fa"
    BTN_FG = "#1e1e2e"
    ENTRY_BG = "#313244"
    LIST_BG = "#181825"
    LIST_SEL = "#45475a"
    HOVER = "#585b70"

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("카카오톡 이모티콘 추출기 (PC)")
        self.root.geometry("820x650")
        self.root.configure(bg=self.BG)
        self.root.minsize(700, 500)

        self.results: list[dict] = []
        self.thumb_images: list = []  # keep references so GC doesn't collect

        self._build_ui()
        self.root.mainloop()

    # ── UI construction ──

    def _build_ui(self):
        # Title
        title_frame = tk.Frame(self.root, bg=self.BG)
        title_frame.pack(fill="x", padx=20, pady=(18, 4))
        tk.Label(
            title_frame, text="🎨 카카오톡 이모티콘 추출기",
            font=("Segoe UI", 20, "bold"), bg=self.BG, fg=self.ACCENT
        ).pack(side="left")

        # Search bar
        search_frame = tk.Frame(self.root, bg=self.BG)
        search_frame.pack(fill="x", padx=20, pady=(8, 4))

        self.search_var = tk.StringVar()
        entry = tk.Entry(
            search_frame, textvariable=self.search_var,
            font=("Segoe UI", 13), bg=self.ENTRY_BG, fg=self.FG,
            insertbackground=self.FG, relief="flat", bd=0
        )
        entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 8))
        entry.bind("<Return>", lambda e: self._on_search())

        search_btn = tk.Button(
            search_frame, text="검색", font=("Segoe UI", 12, "bold"),
            bg=self.BTN_BG, fg=self.BTN_FG, activebackground="#74c7ec",
            relief="flat", cursor="hand2", padx=18, pady=4,
            command=self._on_search
        )
        search_btn.pack(side="right")

        # Status label
        self.status_var = tk.StringVar(value="검색어를 입력하세요.")
        tk.Label(
            self.root, textvariable=self.status_var,
            font=("Segoe UI", 10), bg=self.BG, fg="#a6adc8"
        ).pack(anchor="w", padx=22, pady=(2, 2))

        # Results list
        list_frame = tk.Frame(self.root, bg=self.LIST_BG)
        list_frame.pack(fill="both", expand=True, padx=20, pady=(4, 4))

        columns = ("title", "artist")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("title", text="이모티콘 이름")
        self.tree.heading("artist", text="작가")
        self.tree.column("title", width=420, minwidth=200)
        self.tree.column("artist", width=250, minwidth=100)

        # Style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview",
                        background=self.LIST_BG, foreground=self.FG,
                        fieldbackground=self.LIST_BG, font=("Segoe UI", 11),
                        rowheight=32)
        style.configure("Treeview.Heading",
                        background=self.ENTRY_BG, foreground=self.ACCENT,
                        font=("Segoe UI", 11, "bold"))
        style.map("Treeview",
                  background=[("selected", self.LIST_SEL)],
                  foreground=[("selected", self.ACCENT)])

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)

        # Bottom buttons
        btn_frame = tk.Frame(self.root, bg=self.BG)
        btn_frame.pack(fill="x", padx=20, pady=(4, 14))

        download_btn = tk.Button(
            btn_frame, text="📥 선택 다운로드", font=("Segoe UI", 12, "bold"),
            bg="#a6e3a1", fg=self.BTN_FG, activebackground="#94e2d5",
            relief="flat", cursor="hand2", padx=16, pady=6,
            command=self._on_download
        )
        download_btn.pack(side="left")

        download_all_btn = tk.Button(
            btn_frame, text="📦 전체 다운로드", font=("Segoe UI", 12, "bold"),
            bg="#fab387", fg=self.BTN_FG, activebackground="#eba0ac",
            relief="flat", cursor="hand2", padx=16, pady=6,
            command=self._on_download_all
        )
        download_all_btn.pack(side="left", padx=(10, 0))

        preview_btn = tk.Button(
            btn_frame, text="👀 미리보기", font=("Segoe UI", 12, "bold"),
            bg=self.BTN_BG, fg=self.BTN_FG, activebackground="#74c7ec",
            relief="flat", cursor="hand2", padx=16, pady=6,
            command=self._on_preview
        )
        preview_btn.pack(side="right")

        # Footer
        tk.Label(
            self.root,
            text="이 프로그램은 카카오톡 및 카카오와 관련이 없습니다. 사용으로 인한 책임은 사용자에게 있습니다.\n"
                 "Original Android app © 2021 Dark Tornado (GPL 3.0)",
            font=("Segoe UI", 8), bg=self.BG, fg="#585b70", justify="center"
        ).pack(pady=(0, 8))

    # ── Actions ──

    def _on_search(self):
        query = self.search_var.get().strip()
        if not query:
            messagebox.showwarning("알림", "검색어를 입력하세요.")
            return
        self.status_var.set("검색 중...")
        self.tree.delete(*self.tree.get_children())
        self.results.clear()
        threading.Thread(target=self._search_thread, args=(query,), daemon=True).start()

    def _search_thread(self, query: str):
        try:
            items = search_emoticons(query)
            if not items:
                self.root.after(0, lambda: self.status_var.set("검색 결과가 없습니다."))
                self.root.after(0, lambda: messagebox.showinfo("알림", "검색 결과가 없어요."))
                return
            self.results = items
            self.root.after(0, lambda: self._populate_list(items))
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"검색 실패: {e}"))
            self.root.after(0, lambda: messagebox.showerror("오류", f"이모티콘 검색 실패\n{e}"))

    def _populate_list(self, items: list):
        self.tree.delete(*self.tree.get_children())
        for i, item in enumerate(items):
            title = item.get("title", "?")
            artist = item.get("artist", "?")
            self.tree.insert("", "end", iid=str(i), values=(title, artist))
        self.status_var.set(f"검색 결과: {len(items)}개")

    def _get_selected_index(self) -> int | None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("알림", "이모티콘을 선택하세요.")
            return None
        return int(sel[0])

    def _on_download(self):
        idx = self._get_selected_index()
        if idx is None:
            return
        item = self.results[idx]
        save_dir = filedialog.askdirectory(title="저장할 폴더 선택")
        if not save_dir:
            return
        self.status_var.set("다운로드 중...")
        threading.Thread(target=self._download_thread, args=(item, save_dir), daemon=True).start()

    def _on_download_all(self):
        if not self.results:
            messagebox.showwarning("알림", "먼저 검색을 해주세요.")
            return
        save_dir = filedialog.askdirectory(title="저장할 폴더 선택")
        if not save_dir:
            return
        self.status_var.set("전체 다운로드 중...")
        threading.Thread(target=self._download_all_thread, args=(save_dir,), daemon=True).start()

    def _download_thread(self, item: dict, save_dir: str):
        try:
            title = item.get("title", "emoticon").replace("/", "_").replace("\\", "_")
            title_url = item.get("titleUrl", "")
            thumbnails = get_emoticon_thumbnails(title_url)
            if not thumbnails:
                self.root.after(0, lambda: messagebox.showinfo("알림", "다운로드할 이미지가 없습니다."))
                return

            folder = os.path.join(save_dir, title)
            os.makedirs(folder, exist_ok=True)

            success, fail = 0, 0
            for i, url in enumerate(thumbnails):
                try:
                    ext = self._guess_ext(url)
                    data = fetch_bytes(url)
                    filepath = os.path.join(folder, f"{i}{ext}")
                    with open(filepath, "wb") as f:
                        f.write(data)
                    success += 1
                except Exception:
                    fail += 1

            msg = f"다운로드 완료!\n{len(thumbnails)}개 중 {success}개 성공, {fail}개 실패\n저장 위치: {folder}"
            self.root.after(0, lambda: self.status_var.set(f"다운로드 완료: {title}"))
            self.root.after(0, lambda: messagebox.showinfo("완료", msg))
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"다운로드 실패: {e}"))
            self.root.after(0, lambda: messagebox.showerror("오류", f"다운로드 실패\n{e}"))

    def _download_all_thread(self, save_dir: str):
        total_success, total_fail, total_packs = 0, 0, 0
        for item in self.results:
            try:
                title = item.get("title", "emoticon").replace("/", "_").replace("\\", "_")
                title_url = item.get("titleUrl", "")
                thumbnails = get_emoticon_thumbnails(title_url)
                if not thumbnails:
                    continue

                folder = os.path.join(save_dir, title)
                os.makedirs(folder, exist_ok=True)

                for i, url in enumerate(thumbnails):
                    try:
                        ext = self._guess_ext(url)
                        data = fetch_bytes(url)
                        with open(os.path.join(folder, f"{i}{ext}"), "wb") as f:
                            f.write(data)
                        total_success += 1
                    except Exception:
                        total_fail += 1
                total_packs += 1
                self.root.after(0, lambda t=title: self.status_var.set(f"다운로드 중: {t}"))
            except Exception:
                total_fail += 1

        msg = f"전체 다운로드 완료!\n{total_packs}개 팩, {total_success}개 성공, {total_fail}개 실패\n저장 위치: {save_dir}"
        self.root.after(0, lambda: self.status_var.set("전체 다운로드 완료"))
        self.root.after(0, lambda: messagebox.showinfo("완료", msg))

    def _on_preview(self):
        idx = self._get_selected_index()
        if idx is None:
            return
        item = self.results[idx]
        self.status_var.set("미리보기 로딩 중...")
        threading.Thread(target=self._preview_thread, args=(item,), daemon=True).start()

    def _preview_thread(self, item: dict):
        try:
            title = item.get("title", "?")
            title_url = item.get("titleUrl", "")
            thumbnails = get_emoticon_thumbnails(title_url)
            if not thumbnails:
                self.root.after(0, lambda: messagebox.showinfo("알림", "미리보기 이미지가 없습니다."))
                return

            if not HAS_PIL:
                self.root.after(0, lambda: messagebox.showinfo(
                    "알림",
                    f"미리보기를 사용하려면 Pillow 라이브러리가 필요합니다.\n"
                    f"pip install Pillow\n\n"
                    f"이모티콘 {len(thumbnails)}개 발견됨."
                ))
                self.root.after(0, lambda: self.status_var.set(f"미리보기: {title} ({len(thumbnails)}개)"))
                return

            images_data = []
            for url in thumbnails[:24]:  # limit to 24 for preview
                try:
                    raw = fetch_bytes(url)
                    img = Image.open(io.BytesIO(raw))
                    img = img.resize((80, 80), Image.LANCZOS)
                    images_data.append(img)
                except Exception:
                    pass

            self.root.after(0, lambda: self._show_preview_window(title, images_data, len(thumbnails)))
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"미리보기 실패: {e}"))
            self.root.after(0, lambda: messagebox.showerror("오류", f"미리보기 실패\n{e}"))

    def _show_preview_window(self, title: str, images: list, total: int):
        self.status_var.set(f"미리보기: {title}")
        win = tk.Toplevel(self.root)
        win.title(f"미리보기 - {title}")
        win.configure(bg=self.BG)
        win.geometry("560x480")
        win.resizable(True, True)

        tk.Label(
            win, text=f"🎨 {title}  ({total}개)",
            font=("Segoe UI", 14, "bold"), bg=self.BG, fg=self.ACCENT
        ).pack(pady=(12, 8))

        canvas = tk.Canvas(win, bg=self.LIST_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=self.LIST_BG)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=(0, 12))
        scrollbar.pack(side="right", fill="y", pady=(0, 12))

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        win.protocol("WM_DELETE_WINDOW", lambda: (canvas.unbind_all("<MouseWheel>"), win.destroy()))

        self.thumb_images = []
        cols = 6
        for i, img in enumerate(images):
            tk_img = ImageTk.PhotoImage(img)
            self.thumb_images.append(tk_img)
            row, col = divmod(i, cols)
            lbl = tk.Label(scroll_frame, image=tk_img, bg=self.LIST_BG, bd=1, relief="solid")
            lbl.grid(row=row, column=col, padx=4, pady=4)

        if total > len(images):
            tk.Label(
                scroll_frame,
                text=f"... 외 {total - len(images)}개 더 있음 (다운로드하면 전체 저장)",
                font=("Segoe UI", 9), bg=self.LIST_BG, fg="#a6adc8"
            ).grid(row=(len(images) // cols) + 1, column=0, columnspan=cols, pady=8)

    @staticmethod
    def _guess_ext(url: str) -> str:
        lower = url.lower()
        if ".gif" in lower:
            return ".gif"
        elif ".webp" in lower:
            return ".webp"
        elif ".jpg" in lower or ".jpeg" in lower:
            return ".jpg"
        return ".png"


# ─── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    KakaoEmoticonApp()
