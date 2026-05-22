# ANKEL — 生徒会総務AIアシスタント

> ローカル完結型 AI インフラ。外部 API 不使用。完全プライバシー。

---

## ファイル構成

```
（すべて同じディレクトリに置く）
├── index.html            # UI 構造・エントリーポイント
├── style.css             # デザイン（黒・金・水色）
├── main.js               # UI制御・DB操作・RAG・Worker通信（統合済み）
├── ai-worker.js          # LLM推論（Wllama）+ Embedding（Transformers.js）
├── coi-serviceworker.js  # SharedArrayBuffer 対応（GitHub Pages用）
└── README.md             # このファイル
```

`storage.js` は `main.js` にインライン統合済みのため不要です。

---

## GitHub Pages へのデプロイ手順

1. 上記 **6ファイル**をリポジトリにそのままアップロード（サブディレクトリ不要）
2. Settings → Pages → Branch を設定して公開
3. `coi-serviceworker.js` は **必ずルート（index.html と同じ場所）** に置く

### なぜ coi-serviceworker.js が必要か

Wllama のマルチスレッド推論は `SharedArrayBuffer` を必要とします。  
GitHub Pages はデフォルトで無効ですが、このファイルが  
`Cross-Origin-Embedder-Policy: require-corp` を自動付与して解決します。

---

## 初回利用フロー

```
ページを開く
  → Scrapbox JSON をドロップ（Scrapbox: 設定 → Export）
  → 全ページを IndexedDB にインポート（Embedding も自動生成）
  → モデル選択 → ロード → チャット開始
```

---

## モデル選択

| モード | モデル | 用途 |
|--------|--------|------|
| 標準（校閲・高速） | Llama-3.2-1B-Instruct Q4_K_M | 日常質問・校閲 |
| 文章作成特化 | Qwen2.5-1.5B-Instruct Q4_K_M | 文書・議事録の作成 |
| 論理・解決特化 | Gemma-2-2B-it Q4_K_M | 問題分析・意思決定 |

- 初回のみ HuggingFace からダウンロード（700MB〜1.5GB）
- 2回目以降はブラウザキャッシュで**オフライン動作**

---

## データ管理

- 全データは **ブラウザの IndexedDB** に保存（外部送信なし）
- 再インポートで既存データは上書きされます
- データ削除: DevTools → Application → IndexedDB → `ankel_kb` を削除

---

## Scrapbox タイトル命名規則

```
yyyy-業務タグ-自由記述
例: 2024-会計-前期予算書の修正手順
例: 2023-広報-文化祭ポスター制作フロー
```

この形式に従うとサイドバーの年度・タグフィルタが機能します。

---

## 次世代への引き継ぎ

**モデルを追加する場合**
1. `ai-worker.js` の `MODELS` オブジェクトに HuggingFace の GGUF URL を追加
2. `index.html` の `<select id="model-select">` に同じ `value` で `<option>` を追加

**カラーテーマを変更する場合**
- `style.css` の `:root` 内の `--gold` / `--blue` 変数を書き換えるだけ

---

*ANKEL — Built for continuity. Runs locally. Remembers everything.*
