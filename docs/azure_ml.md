---
title: Azure ML SDK v2 Playground チュートリアル
description: playground_azureml.py を使ってコマンドジョブ投入からステータス確認までを手を動かして体験する再現手順書
author: ks6088ts-labs
ms.date: 2026-03-17
ms.topic: tutorial
keywords:
  - azure machine learning
  - azure ml sdk v2
  - python
  - tutorial
---

## 概要

`scripts/playground_azureml.py` は Azure ML SDK v2 の基本操作を段階的に体験できる CLI ツールです。
この手順書では、環境セットアップからコマンドジョブの投入・監視・出力 URI の確認までを順に再現できます。

| フェーズ                  | サブコマンド              | Azure 接続 |
|---------------------------|---------------------------|------------|
| 1. 環境確認               | `sdk-version`             | 不要       |
| 2. 接続設定確認           | `show-config`             | 不要       |
| 3. リソース確認           | `list-computes`           | 必要       |
| 4. ジョブ投入（dry-run）  | `submit-command-job`      | 不要       |
| 5. ジョブ投入（実行）     | `submit-command-job`      | 必要       |
| 6. ジョブ監視             | `get-job`                 | 必要       |
| 7. パイプライン動作確認   | `demo-dsl-pipeline`       | 不要       |

## 前提条件

* Python 3.10 以上
* `uv` コマンドが利用可能であること
* 依存パッケージをインストール済みであること

```bash
uv sync
```

## Phase 1 — 環境確認

### 1-1. SDK バージョンを確認する

インストール済みの `azure-ai-ml` バージョンと、SDK v1 → v2 の主要 API 差分テーブルを表示します。

```bash
uv run scripts/playground_azureml.py sdk-version
```

出力に `azure-ai-ml version :` の行が表示されれば SDK のインストールは正常です。

```text
azure-ai-ml version : 1.32.0
SDK family          : v2  (azure-ai-ml ≥ 1.0)
Support status      : Active (v1 azureml-sdk EOL 2026-06-30)

+-------------------+---------------------------------------------+---------------------------------------------------+
| Concept           | SDK v1                                      | SDK v2                                            |
+-------------------+---------------------------------------------+---------------------------------------------------+
| Workspace connect | Workspace.from_config()                     | MLClient(DefaultAzureCredential(), ...)           |
| Job submit        | experiment.submit(ScriptRunConfig)          | ml_client.jobs.create_or_update(job)              |
| Job status        | run.get_status()                            | job.status  (property, no polling needed)         |
| Job stream        | run.wait_for_completion()                   | ml_client.jobs.stream(job.name)                   |
| Metric logging    | run.log('acc', 0.95)                        | mlflow.log_metric('acc', 0.95)                    |
| Model register    | Model.register(ws, ...)                     | ml_client.models.create_or_update(Model(...))     |
+-------------------+---------------------------------------------+---------------------------------------------------+
```

## Phase 2 — 接続設定確認

### 2-1. .env ファイルを作成する

プロジェクトルートに `.env` ファイルを作成し、Azure リソースの情報を記載します。

```bash
cat > .env <<'EOF'
AZURE_SUBSCRIPTION_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
AZURE_RESOURCE_GROUP=my-resource-group
AZURE_WORKSPACE_NAME=my-aml-workspace
EOF
```

### 2-2. 接続設定を表示する

`.env` が正しく読み込まれているかを確認します。Azure への接続は行いません。

```bash
uv run scripts/playground_azureml.py show-config
```

`AZURE_SUBSCRIPTION_ID` がマスク表示（`xxxx****xxxx` 形式）されれば読み込み成功です。

```text
Azure ML connection settings:
  AZURE_SUBSCRIPTION_ID          = xxxx****xxxx
  AZURE_RESOURCE_GROUP           = my-resource-group
  AZURE_WORKSPACE_NAME           = my-aml-workspace

✅ All required connection settings are present.
```

3 項目すべてが `(not set)` の場合は `.env` の配置場所またはキー名を確認してください。

```text
[warn] 3 setting(s) not configured: AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE_NAME
       Set them in .env or as environment variables before running subsequent steps.
```

## Phase 3 — リソース確認

ジョブ投入前に、ターゲットとなるコンピュートリソースが存在することを確認します。

### 3-1. コンピュートリソースを一覧表示する

```bash
uv run scripts/playground_azureml.py list-computes
```

```text
Compute resources:

  cpu-cluster                         type=amlcompute           state=Succeeded
  gpu-cluster                         type=amlcompute           state=Succeeded

2 compute resource(s) found.
```

`cpu-cluster` が表示されることを確認してください。表示されない場合は Azure ML Studio でクラスターを作成するか、`--compute` オプションで別のクラスター名を指定してください。

> [!TIP]
> ワークスペース内の他のリソースも同様に確認できます。
>
> ```bash
> uv run scripts/playground_azureml.py list-workspaces
> uv run scripts/playground_azureml.py list-datastores
> uv run scripts/playground_azureml.py list-environments --max-results 5
> uv run scripts/playground_azureml.py list-models
> uv run scripts/playground_azureml.py list-jobs --max-results 5
> ```

## Phase 4 — ジョブ投入（dry-run）

実際に Azure に送信する前に、`--dry-run` フラグでジョブ定義を手元で確認します。

```bash
uv run scripts/playground_azureml.py submit-command-job --dry-run
```

```text
[dry-run] Job definition (not submitted):

  display_name : playground-echo
  command      : echo 'Hello from Azure ML SDK v2 playground!' && echo 'compute: $AZUREML_COMPUTE_CLUSTER_NAME'
  compute      : cpu-cluster
  environment  : Environment({'arm_type': 'environment_version', ..., 'image': 'mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest', ...})
  experiment   : playground-azureml-tutorial
```

`display_name`・`command`・`compute` が意図どおりであれば次フェーズへ進みます。

異なるコンピュートや実験名を使う場合はオプションで上書きできます。

```bash
uv run scripts/playground_azureml.py submit-command-job \
  --dry-run \
  --compute my-compute \
  --experiment my-experiment
```

## Phase 5 — ジョブ投入（実行）

dry-run で内容を確認したら、`--dry-run` を外して実際に投入します。

```bash
uv run scripts/playground_azureml.py submit-command-job \
  --compute cpu-cluster \
  --experiment playground-azureml-tutorial
```

投入に成功すると、ジョブ名と Studio URL が表示されます。

```text
Submitting job to experiment 'playground-azureml-tutorial' on compute 'cpu-cluster'…

✅ Job submitted successfully.
   name        : brave_cup_abc123xyz
   status      : Starting
   studio url  : https://ml.azure.com/runs/brave_cup_abc123xyz?...

To tail logs:  uv run scripts/playground_azureml.py get-job --job-name brave_cup_abc123xyz
```

> [!IMPORTANT]
> 表示された `name` の値（例: `brave_cup_abc123xyz`）を次フェーズで使用します。
> ターミナルに残しておくか、変数に保存してください。
>
> ```bash
> JOB_NAME=brave_cup_abc123xyz
> ```

### submit-command-job オプション一覧

| オプション     | 短縮形 | デフォルト値                    | 説明                         |
|----------------|--------|---------------------------------|------------------------------|
| `--compute`    | `-c`   | `cpu-cluster`                   | コンピュートクラスタ名       |
| `--experiment` | `-e`   | `playground-azureml-tutorial`   | 実験名                       |
| `--dry-run`    |        | `False`                         | 投入せずにジョブ定義を表示   |
| `--verbose`    | `-v`   | `False`                         | デバッグログを有効化         |

## Phase 6 — ジョブ監視

### 6-1. ジョブステータスを確認する

投入直後はステータスが `Starting` や `Queued` であることが多いです。
以下のコマンドを繰り返してステータス変化を追います。

```bash
uv run scripts/playground_azureml.py get-job --job-name $JOB_NAME
```

実行中の出力例（ステータス: `Running`）:

```text
Job: brave_cup_abc123xyz
  display_name  : playground-echo
  status        : ⏳ Running
  experiment    : playground-azureml-tutorial

Output URI resolution for 'default':
  [info] Job is in 'Running' state — output URIs are not yet finalised.
         Output URI format (predictable before completion):
         azureml://datastores/workspaceblobstore/paths/azureml/brave_cup_abc123xyz/default/
```

完了後の出力例（ステータス: `Completed`）:

```text
Job: brave_cup_abc123xyz
  display_name  : playground-echo
  status        : ✅ Completed
  experiment    : playground-azureml-tutorial
  studio_url    : https://ml.azure.com/runs/brave_cup_abc123xyz?...

Output URI resolution for 'default':
  resolved URI  : azureml://datastores/workspaceblobstore/paths/azureml/brave_cup_abc123xyz/default/

To download all outputs:
  ml_client.jobs.download(name='brave_cup_abc123xyz', download_path='./outputs', all=True)
```

### ジョブステータスの遷移

| ステータス     | 分類         | 指標 |
|----------------|--------------|------|
| `NotStarted`   | 実行中       | ⏳   |
| `Starting`     | 実行中       | ⏳   |
| `Provisioning` | 実行中       | ⏳   |
| `Preparing`    | 実行中       | ⏳   |
| `Queued`       | 実行中       | ⏳   |
| `Running`      | 実行中       | ⏳   |
| `Finalizing`   | 実行中       | ⏳   |
| `Completed`    | 完了（成功） | ✅   |
| `Failed`       | 完了（失敗） | ❌   |
| `Canceled`     | 完了（中断） | ❌   |

### 6-2. 出力ファイルをダウンロードする

ステータスが `Completed` になったら Python スクリプト内で出力をダウンロードできます。

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="<AZURE_SUBSCRIPTION_ID>",
    resource_group_name="<AZURE_RESOURCE_GROUP>",
    workspace_name="<AZURE_WORKSPACE_NAME>",
)

ml_client.jobs.download(
    name="brave_cup_abc123xyz",
    download_path="./outputs",
    all=True,
)
```

### 6-3. 過去のジョブを一覧で確認する

実験名でフィルタリングすると投入した一連のジョブをまとめて参照できます。

```bash
uv run scripts/playground_azureml.py list-jobs \
  --experiment playground-azureml-tutorial \
  --max-results 10
```

## Phase 7 — パイプラインの動作確認（ローカル）

`@pipeline` DSL を使ったパイプライン定義をローカルで確認します。Azure への接続は不要です。

### 7-1. デフォルトパラメータで実行する

```bash
uv run scripts/playground_azureml.py demo-dsl-pipeline
```

```text
======================================================================
DSL @pipeline with typed parameters
======================================================================

  Pipeline inputs passed at instantiation time:
    n_epoch       = 3  (int  → integer)
    backbone      = 'effb3'  (str  → string)
    learning_rate = 0.001  (float → number)

  Resolved pipeline_job.inputs (type → internal representation):
    n_epoch            = 3  (int)
    backbone           = 'effb3'  (str)
    learning_rate      = 0.001  (float)

  [dry-run] Pipeline built successfully.  Pass --no-dry-run to submit.

======================================================================
```

### 7-2. パラメータを変えて実行する

```bash
uv run scripts/playground_azureml.py demo-dsl-pipeline \
  --n-epoch 10 \
  --backbone effb4 \
  --lr 5e-4
```

### 7-3. パイプラインを Azure ML に投入する

接続設定が整っている場合は `--no-dry-run` を付けて実際に投入できます。

```bash
uv run scripts/playground_azureml.py demo-dsl-pipeline --no-dry-run
```

### パイプライン入力の型対応

`@pipeline` 関数の引数に型アノテーションを付けることで、SDK が内部で自動的に正しい `LiteralJobInput` 型に変換します。
型ごとの正しい代入方法は次のとおりです。

| YAML `type:`  | 正しい代入方法                                                             |
|---------------|----------------------------------------------------------------------------|
| `integer`     | `pipeline_job.inputs['key'] = 3`                                           |
| `number`      | `pipeline_job.inputs['key'] = 0.001`                                       |
| `string`      | `pipeline_job.inputs['key'] = 'value'`                                     |
| `boolean`     | `pipeline_job.inputs['key'] = True`                                        |
| `uri_folder`  | `pipeline_job.inputs['key'] = Input(type=AssetTypes.URI_FOLDER, path=...)` |
| `uri_file`    | `pipeline_job.inputs['key'] = Input(type=AssetTypes.URI_FILE, path=...)`   |

> [!NOTE]
> `Input(type='integer', ...)` のように `Input()` をプリミティブ型に使うと、SDK 内部のシリアライザが `LiteralJobInput` への変換に失敗します。
> プリミティブ型（integer / number / string / boolean）には Python スカラーを直接代入してください。
> `Input()` はデータ・モデルアセット（`uri_folder` / `uri_file` / `mltable`）専用です。

## ヘルプの確認

各サブコマンドのオプション一覧は `--help` で確認できます。

```bash
# 全サブコマンド一覧
uv run scripts/playground_azureml.py --help

# 個別のヘルプ
uv run scripts/playground_azureml.py submit-command-job --help
uv run scripts/playground_azureml.py get-job --help
```

## 参考リンク

* [Azure Machine Learning ドキュメント](https://learn.microsoft.com/azure/machine-learning/concept-azure-machine-learning-v2)
* [Azure ML SDK v2 — パイプライン入出力の管理](https://learn.microsoft.com/azure/machine-learning/how-to-manage-inputs-outputs-pipeline)
* [SDK v1 から v2 への移行ガイド](https://learn.microsoft.com/azure/machine-learning/how-to-migrate-from-v1)
