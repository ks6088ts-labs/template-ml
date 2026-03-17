"""Azure ML SDK v2 Playground — step-by-step tutorial CLI.

Subcommand map (run each in order for a guided walkthrough):

  Step 0  sdk-version               Show installed SDK version + v1→v2 migration notes.
  Step 1  show-config               Display connection settings from environment variables.
  Step 2  list-workspaces           List Azure ML workspaces in a subscription.
  Step 3  list-computes             List compute resources in a workspace.
  Step 4  list-datastores           List datastores in a workspace.
  Step 5  list-environments         List environments in a workspace.
  Step 6  list-models               List registered models in a workspace.
  Step 7  list-jobs                 List recent jobs (optionally filtered by experiment).
  Step 8  submit-command-job        Submit a minimal echo command job.
  Step 9  get-job                   Get job status + resolved output URI.
  Step 10 demo-dsl-pipeline         DSL @pipeline with typed kwargs (local only).

Environment variables (can also be placed in a .env file):

  AZURE_SUBSCRIPTION_ID   Azure subscription ID
  AZURE_RESOURCE_GROUP    Resource group that contains the workspace
  AZURE_WORKSPACE_NAME    Azure ML workspace name

ref. https://learn.microsoft.com/azure/machine-learning/concept-azure-machine-learning-v2
"""

import importlib.metadata
import logging
from typing import Annotated

import typer
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

from template_ml.loggers import get_logger

# ---------------------------------------------------------------------------
# App & logger
# ---------------------------------------------------------------------------

app = typer.Typer(
    add_completion=False,
    help="Azure ML SDK v2 Playground — step-by-step tutorial CLI.",
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class AzureMlSettings(BaseSettings):
    """Azure ML connection settings.  Override any field via env var or .env file."""

    azure_subscription_id: str = ""
    azure_resource_group: str = ""
    azure_workspace_name: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _set_verbose(verbose: bool) -> None:
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)


def _resolve(cli_value: str | None, setting_value: str, env_name: str) -> str:
    """Return the first non-empty value from CLI → settings → abort."""
    resolved = cli_value or setting_value
    if not resolved:
        typer.echo(
            f"[error] Required value missing: pass --{env_name.lower().replace('_', '-')} "
            f"or set AZURE_{env_name.upper()} in the environment / .env file.",
            err=True,
        )
        raise typer.Exit(code=1)
    return resolved


def _get_ml_client(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
):
    """Build an authenticated MLClient with DefaultAzureCredential → InteractiveBrowserCredential fallback."""
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

    try:
        credential = DefaultAzureCredential()
        # Eagerly probe reachability so we fall back fast when running locally without
        # a managed identity or cached token.
        credential.get_token("https://management.azure.com/.default")
        logger.debug("Authenticated via DefaultAzureCredential")
    except Exception as exc:
        logger.debug("DefaultAzureCredential failed (%s); falling back to InteractiveBrowserCredential", exc)
        credential = InteractiveBrowserCredential()

    return MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )


# ---------------------------------------------------------------------------
# Shared option type aliases
# ---------------------------------------------------------------------------

_SubscriptionOpt = Annotated[
    str | None,
    typer.Option("--subscription-id", "-s", help="Azure subscription ID (or AZURE_SUBSCRIPTION_ID env var)"),
]
_ResourceGroupOpt = Annotated[
    str | None,
    typer.Option("--resource-group", "-g", help="Resource group name (or AZURE_RESOURCE_GROUP env var)"),
]
_WorkspaceOpt = Annotated[
    str | None,
    typer.Option("--workspace", "-w", help="Azure ML workspace name (or AZURE_WORKSPACE_NAME env var)"),
]
_VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging")]


# ===========================================================================
# Step 0 — sdk-version
# ===========================================================================


@app.command(
    name="sdk-version",
    help=("Step 0 — Show installed azure-ai-ml version and SDK v1 → v2 migration notes."),
)
def sdk_version(
    verbose: _VerboseOpt = False,
) -> None:
    """Display SDK version metadata and key v1 → v2 API differences."""
    _set_verbose(verbose)

    try:
        version = importlib.metadata.version("azure-ai-ml")
    except importlib.metadata.PackageNotFoundError:
        typer.echo("[error] azure-ai-ml is not installed.  Run: uv add azure-ai-ml", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"\nazure-ai-ml version : {version}")
    typer.echo("SDK family          : v2  (azure-ai-ml ≥ 1.0)")
    typer.echo("Support status      : Active (v1 azureml-sdk EOL 2026-06-30)\n")

    table = [
        ("Concept", "SDK v1", "SDK v2"),
        ("Workspace connect", "Workspace.from_config()", "MLClient(DefaultAzureCredential(), ...)"),
        ("Job submit", "experiment.submit(ScriptRunConfig)", "ml_client.jobs.create_or_update(job)"),
        ("Job status", "run.get_status()", "job.status  (property, no polling needed)"),
        ("Job stream", "run.wait_for_completion()", "ml_client.jobs.stream(job.name)"),
        ("Metric logging", "run.log('acc', 0.95)", "mlflow.log_metric('acc', 0.95)"),
        ("Model register", "Model.register(ws, ...)", "ml_client.models.create_or_update(Model(...))"),
    ]

    col_widths = [max(len(row[i]) for row in table) + 2 for i in range(3)]
    sep = "+" + "+".join("-" * w for w in col_widths) + "+"

    typer.echo(sep)
    for i, row in enumerate(table):
        line = "|" + "|".join(f" {cell:<{col_widths[j] - 1}}" for j, cell in enumerate(row)) + "|"
        typer.echo(line)
        if i == 0:
            typer.echo(sep)
    typer.echo(sep)
    typer.echo()


# ===========================================================================
# Step 1 — show-config
# ===========================================================================


@app.command(
    name="show-config",
    help="Step 1 — Display Azure ML connection settings from environment variables / .env file.",
)
def show_config(
    verbose: _VerboseOpt = False,
) -> None:
    """Print resolved connection settings without establishing an actual network connection."""
    _set_verbose(verbose)

    settings = AzureMlSettings()

    def _mask(value: str) -> str:
        return value[:4] + "****" + value[-4:] if len(value) > 8 else "****"

    rows = {
        "AZURE_SUBSCRIPTION_ID": _mask(settings.azure_subscription_id)
        if settings.azure_subscription_id
        else "(not set)",
        "AZURE_RESOURCE_GROUP": settings.azure_resource_group or "(not set)",
        "AZURE_WORKSPACE_NAME": settings.azure_workspace_name or "(not set)",
    }

    typer.echo("\nAzure ML connection settings:")
    for key, val in rows.items():
        typer.echo(f"  {key:<30} = {val}")
    typer.echo()

    missing = [k for k, v in rows.items() if v == "(not set)"]
    if missing:
        typer.echo(f"[warn] {len(missing)} setting(s) not configured: {', '.join(missing)}")
        typer.echo("       Set them in .env or as environment variables before running subsequent steps.\n")
    else:
        typer.echo("✅ All required connection settings are present.\n")


# ===========================================================================
# Step 2 — list-workspaces
# ===========================================================================


@app.command(
    name="list-workspaces",
    help="Step 2 — List Azure ML workspaces accessible within a subscription.",
)
def list_workspaces(
    subscription_id: _SubscriptionOpt = None,
    resource_group: _ResourceGroupOpt = None,
    verbose: _VerboseOpt = False,
) -> None:
    """List all workspaces in the resolved subscription (resource_group is optional here)."""
    _set_verbose(verbose)

    from azure.ai.ml import MLClient
    from azure.ai.ml.constants import Scope
    from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

    settings = AzureMlSettings()
    sub_id = _resolve(subscription_id, settings.azure_subscription_id, "SUBSCRIPTION_ID")
    rg = resource_group or settings.azure_resource_group or None

    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as exc:
        logger.debug("DefaultAzureCredential unavailable (%s), falling back", exc)
        credential = InteractiveBrowserCredential()

    # Use a subscription-scoped client (no workspace_name required)
    ml_client = MLClient(credential=credential, subscription_id=sub_id, resource_group_name=rg or "")

    scope = Scope.RESOURCE_GROUP if rg else Scope.SUBSCRIPTION
    typer.echo(f"\nListing workspaces (scope={scope}, subscription={sub_id[:8]}…):\n")

    count = 0
    for ws in ml_client.workspaces.list(scope=scope):
        count += 1
        typer.echo(f"  {ws.name:<40} location={ws.location}  rg={ws.resource_group}")

    typer.echo(f"\n{count} workspace(s) found.\n")


# ===========================================================================
# Step 3 — list-computes
# ===========================================================================


@app.command(
    name="list-computes",
    help="Step 3 — List compute resources (clusters, instances, serverless) in the workspace.",
)
def list_computes(
    subscription_id: _SubscriptionOpt = None,
    resource_group: _ResourceGroupOpt = None,
    workspace: _WorkspaceOpt = None,
    verbose: _VerboseOpt = False,
) -> None:
    _set_verbose(verbose)
    settings = AzureMlSettings()
    ml_client = _get_ml_client(
        _resolve(subscription_id, settings.azure_subscription_id, "SUBSCRIPTION_ID"),
        _resolve(resource_group, settings.azure_resource_group, "RESOURCE_GROUP"),
        _resolve(workspace, settings.azure_workspace_name, "WORKSPACE_NAME"),
    )

    typer.echo("\nCompute resources:\n")
    count = 0
    for compute in ml_client.compute.list():
        count += 1
        compute_type = getattr(compute, "type", "unknown")
        state = getattr(compute, "provisioning_state", "")
        typer.echo(f"  {compute.name:<35} type={compute_type:<20} state={state}")

    typer.echo(f"\n{count} compute resource(s) found.\n")


# ===========================================================================
# Step 4 — list-datastores
# ===========================================================================


@app.command(
    name="list-datastores",
    help="Step 4 — List datastores registered in the workspace.",
)
def list_datastores(
    subscription_id: _SubscriptionOpt = None,
    resource_group: _ResourceGroupOpt = None,
    workspace: _WorkspaceOpt = None,
    verbose: _VerboseOpt = False,
) -> None:
    _set_verbose(verbose)
    settings = AzureMlSettings()
    ml_client = _get_ml_client(
        _resolve(subscription_id, settings.azure_subscription_id, "SUBSCRIPTION_ID"),
        _resolve(resource_group, settings.azure_resource_group, "RESOURCE_GROUP"),
        _resolve(workspace, settings.azure_workspace_name, "WORKSPACE_NAME"),
    )

    typer.echo("\nDatastores:\n")
    count = 0
    for ds in ml_client.datastores.list():
        count += 1
        ds_type = getattr(ds, "type", "unknown")
        is_default = " [default]" if getattr(ds, "is_default", False) else ""
        storage_account = getattr(ds, "account_name", "")
        typer.echo(f"  {ds.name:<35} type={str(ds_type):<25} account={storage_account}{is_default}")

    typer.echo(f"\n{count} datastore(s) found.\n")


# ===========================================================================
# Step 5 — list-environments
# ===========================================================================


@app.command(
    name="list-environments",
    help="Step 5 — List curated and custom environments available in the workspace.",
)
def list_environments(
    subscription_id: _SubscriptionOpt = None,
    resource_group: _ResourceGroupOpt = None,
    workspace: _WorkspaceOpt = None,
    max_results: Annotated[
        int, typer.Option("--max-results", "-n", help="Maximum number of environments to show")
    ] = 20,
    verbose: _VerboseOpt = False,
) -> None:
    _set_verbose(verbose)
    settings = AzureMlSettings()
    ml_client = _get_ml_client(
        _resolve(subscription_id, settings.azure_subscription_id, "SUBSCRIPTION_ID"),
        _resolve(resource_group, settings.azure_resource_group, "RESOURCE_GROUP"),
        _resolve(workspace, settings.azure_workspace_name, "WORKSPACE_NAME"),
    )

    typer.echo(f"\nEnvironments (first {max_results}):\n")
    count = 0
    for env in ml_client.environments.list():
        if count >= max_results:
            break
        count += 1
        version = getattr(env, "version", "?")
        typer.echo(f"  {env.name:<50} version={version}")

    typer.echo(f"\n{count} environment(s) shown (--max-results={max_results}).\n")


# ===========================================================================
# Step 6 — list-models
# ===========================================================================


@app.command(
    name="list-models",
    help="Step 6 — List registered models in the workspace.",
)
def list_models(
    subscription_id: _SubscriptionOpt = None,
    resource_group: _ResourceGroupOpt = None,
    workspace: _WorkspaceOpt = None,
    max_results: Annotated[int, typer.Option("--max-results", "-n", help="Maximum number of models to show")] = 20,
    verbose: _VerboseOpt = False,
) -> None:
    _set_verbose(verbose)
    settings = AzureMlSettings()
    ml_client = _get_ml_client(
        _resolve(subscription_id, settings.azure_subscription_id, "SUBSCRIPTION_ID"),
        _resolve(resource_group, settings.azure_resource_group, "RESOURCE_GROUP"),
        _resolve(workspace, settings.azure_workspace_name, "WORKSPACE_NAME"),
    )

    typer.echo(f"\nRegistered models (first {max_results}):\n")
    count = 0
    for model in ml_client.models.list():
        if count >= max_results:
            break
        count += 1
        version = getattr(model, "version", "?")
        model_type = getattr(model, "type", "")
        typer.echo(f"  {model.name:<45} version={version:<6} type={model_type}")

    typer.echo(f"\n{count} model(s) shown (--max-results={max_results}).\n")


# ===========================================================================
# Step 7 — list-jobs
# ===========================================================================


@app.command(
    name="list-jobs",
    help="Step 7 — List recent jobs in the workspace, optionally filtered by experiment name.",
)
def list_jobs(
    subscription_id: _SubscriptionOpt = None,
    resource_group: _ResourceGroupOpt = None,
    workspace: _WorkspaceOpt = None,
    experiment_name: Annotated[
        str | None,
        typer.Option("--experiment", "-e", help="Filter jobs by experiment name"),
    ] = None,
    max_results: Annotated[int, typer.Option("--max-results", "-n", help="Maximum number of jobs to show")] = 20,
    verbose: _VerboseOpt = False,
) -> None:
    _set_verbose(verbose)
    settings = AzureMlSettings()
    ml_client = _get_ml_client(
        _resolve(subscription_id, settings.azure_subscription_id, "SUBSCRIPTION_ID"),
        _resolve(resource_group, settings.azure_resource_group, "RESOURCE_GROUP"),
        _resolve(workspace, settings.azure_workspace_name, "WORKSPACE_NAME"),
    )

    label = f"experiment='{experiment_name}'" if experiment_name else "all experiments"
    typer.echo(f"\nJobs ({label}, first {max_results}):\n")

    count = 0
    kwargs = {}
    if experiment_name:
        kwargs["parent_job_name"] = None  # list top-level jobs only
    for job in ml_client.jobs.list(**kwargs):
        if experiment_name and getattr(job, "experiment_name", None) != experiment_name:
            continue
        if count >= max_results:
            break
        count += 1
        status = getattr(job, "status", "?")
        exp = getattr(job, "experiment_name", "")
        typer.echo(f"  {job.name:<45} status={status:<15} experiment={exp}")

    typer.echo(f"\n{count} job(s) shown.\n")


# ===========================================================================
# Step 8 — submit-command-job
# ===========================================================================


@app.command(
    name="submit-command-job",
    help="Step 8 — Submit a minimal echo command job to verify end-to-end connectivity.",
)
def submit_command_job(
    subscription_id: _SubscriptionOpt = None,
    resource_group: _ResourceGroupOpt = None,
    workspace: _WorkspaceOpt = None,
    compute: Annotated[
        str,
        typer.Option("--compute", "-c", help="Name of the compute cluster to run on"),
    ] = "cpu-cluster",
    experiment_name: Annotated[
        str,
        typer.Option("--experiment", "-e", help="Experiment name for grouping jobs"),
    ] = "playground-azureml-tutorial",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the job definition without submitting it"),
    ] = False,
    verbose: _VerboseOpt = False,
) -> None:
    """Submit a lightweight command job that just echoes a greeting.

    Use --dry-run to inspect the job definition without touching Azure.
    """
    _set_verbose(verbose)

    from azure.ai.ml import command
    from azure.ai.ml.entities import Environment

    job = command(
        display_name="playground-echo",
        description="Minimal echo job created by playground_azureml.py tutorial.",
        command="echo 'Hello from Azure ML SDK v2 playground!' && echo 'compute: $AZUREML_COMPUTE_CLUSTER_NAME'",
        environment=Environment(
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        ),
        compute=compute,
    )

    if dry_run:
        typer.echo("\n[dry-run] Job definition (not submitted):\n")
        typer.echo(f"  display_name : {job.display_name}")
        typer.echo(f"  command      : {job.command}")
        typer.echo(f"  compute      : {job.compute}")
        typer.echo(f"  environment  : {job.environment}")
        typer.echo(f"  experiment   : {experiment_name}\n")
        return

    settings = AzureMlSettings()
    ml_client = _get_ml_client(
        _resolve(subscription_id, settings.azure_subscription_id, "SUBSCRIPTION_ID"),
        _resolve(resource_group, settings.azure_resource_group, "RESOURCE_GROUP"),
        _resolve(workspace, settings.azure_workspace_name, "WORKSPACE_NAME"),
    )

    typer.echo(f"\nSubmitting job to experiment '{experiment_name}' on compute '{compute}'…")
    returned_job = ml_client.jobs.create_or_update(job, experiment_name=experiment_name)

    typer.echo("\n✅ Job submitted successfully.")
    typer.echo(f"   name        : {returned_job.name}")
    typer.echo(f"   status      : {returned_job.status}")
    studio_url = (returned_job.services or {}).get("Studio")
    if studio_url:
        endpoint = getattr(studio_url, "endpoint", studio_url)
        typer.echo(f"   studio url  : {endpoint}")
    typer.echo(f"\nTo tail logs:  uv run scripts/playground_azureml.py get-job --job-name {returned_job.name}\n")


# ===========================================================================
# Step 9 — get-job
# ===========================================================================


def _resolve_output_uri(ml_client, run_id: str, output_name: str) -> str | None:
    """Return the resolved output URI for *output_name*, or None if it cannot be determined yet.

    Algorithm:
      1. Try the parent job's outputs dict.
      2. If the path is a placeholder (starts with '${{'), enumerate child jobs.
      3. Return None if the job has not completed yet or no URI is available.
    """
    parent = ml_client.jobs.get(run_id)
    parent_outputs = getattr(parent, "outputs", None) or {}
    raw = parent_outputs.get(output_name)
    uri = getattr(raw, "path", None) or getattr(raw, "uri", None)
    if uri and not str(uri).startswith("${{"):
        return uri

    # Walk child jobs of a pipeline
    for child_stub in ml_client.jobs.list(parent_job_name=run_id):
        child = ml_client.jobs.get(child_stub.name)
        child_outputs = getattr(child, "outputs", None) or {}
        child_raw = child_outputs.get(output_name)
        child_uri = getattr(child_raw, "path", None) or getattr(child_raw, "uri", None)
        if child_uri and not str(child_uri).startswith("${{"):
            return child_uri

    return None


@app.command(
    name="get-job",
    help=("Step 9 — Get job status and resolved output URI."),
)
def get_job(
    job_name: Annotated[
        str,
        typer.Option("--job-name", "-j", help="Job name (the run_id returned by submit-command-job)"),
    ],
    output_name: Annotated[
        str,
        typer.Option("--output-name", "-o", help="Name of the output to resolve the URI for"),
    ] = "default",
    subscription_id: _SubscriptionOpt = None,
    resource_group: _ResourceGroupOpt = None,
    workspace: _WorkspaceOpt = None,
    verbose: _VerboseOpt = False,
) -> None:
    """Retrieve and display full job metadata.

    Demonstrates the two-step output URI resolution strategy (parent → child job fallback).
    """
    _set_verbose(verbose)
    settings = AzureMlSettings()
    ml_client = _get_ml_client(
        _resolve(subscription_id, settings.azure_subscription_id, "SUBSCRIPTION_ID"),
        _resolve(resource_group, settings.azure_resource_group, "RESOURCE_GROUP"),
        _resolve(workspace, settings.azure_workspace_name, "WORKSPACE_NAME"),
    )

    job = ml_client.jobs.get(job_name)

    # ---- Status ----
    terminal_statuses = {"Completed", "Failed", "Canceled"}
    in_progress_statuses = {"NotStarted", "Starting", "Provisioning", "Preparing", "Queued", "Running", "Finalizing"}

    status = job.status or "Unknown"
    if status in terminal_statuses:
        indicator = "✅" if status == "Completed" else "❌"
    elif status in in_progress_statuses:
        indicator = "⏳"
    else:
        indicator = "❔"

    typer.echo(f"\nJob: {job.name}")
    typer.echo(f"  display_name  : {getattr(job, 'display_name', '')}")
    typer.echo(f"  status        : {indicator} {status}")
    typer.echo(f"  experiment    : {getattr(job, 'experiment_name', '')}")

    error = getattr(job, "error", None)
    if error:
        typer.echo(f"  error         : {error}")

    studio = (getattr(job, "services", None) or {}).get("Studio")
    if studio:
        endpoint = getattr(studio, "endpoint", studio)
        typer.echo(f"  studio_url    : {endpoint}")

    # ---- Output URI ----
    typer.echo(f"\nOutput URI resolution for '{output_name}':")
    if status != "Completed":
        typer.echo(f"  [info] Job is in '{status}' state — output URIs are not yet finalised.")
        typer.echo(
            "         Output URI format (predictable before completion):\n"
            f"         azureml://datastores/workspaceblobstore/paths/azureml/{job.name}/{output_name}/"
        )
    else:
        uri = _resolve_output_uri(ml_client, job_name, output_name)
        if uri:
            typer.echo(f"  resolved URI  : {uri}")
        else:
            typer.echo(
                f"  [warn] Could not resolve URI for output '{output_name}'. "
                "Check the output name or use the Studio URL above."
            )

    # ---- Download hint ----
    typer.echo(
        f"\nTo download all outputs:\n"
        f"  ml_client.jobs.download(name='{job_name}', download_path='./outputs', all=True)\n"
    )


# ===========================================================================
# Step 10 — demo-dsl-pipeline
# ===========================================================================


@app.command(
    name="demo-dsl-pipeline",
    help=(
        "Step 10 — Demonstrate the @pipeline DSL approach with typed parameters"
        " (local only, no Azure connection required)."
    ),
)
def demo_dsl_pipeline(
    n_epoch: Annotated[int, typer.Option("--n-epoch", help="Number of training epochs")] = 3,
    backbone: Annotated[str, typer.Option("--backbone", help="Model backbone name")] = "effb3",
    learning_rate: Annotated[float, typer.Option("--lr", help="Learning rate")] = 1e-3,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Build pipeline but do not submit")] = True,
    verbose: _VerboseOpt = False,
) -> None:
    """Build and (optionally) submit a DSL-defined pipeline with strongly-typed parameters.

    Because parameters are declared as typed function arguments on the @pipeline-decorated
    function, the SDK infers the correct LiteralJobInput representation automatically —
    no manual coercion required.

    Use --no-dry-run together with the connection options to submit to Azure ML.
    """
    _set_verbose(verbose)

    from azure.ai.ml import Input, Output, command
    from azure.ai.ml.dsl import pipeline

    # Build a lightweight training component inline
    def _make_train_component(compute: str = "cpu-cluster"):
        return command(
            name="train_step",
            display_name="Train model",
            description="Placeholder train step — replace with your real training component.",
            command=(
                "echo 'n_epoch=${{inputs.n_epoch}} backbone=${{inputs.backbone}} lr=${{inputs.learning_rate}}'"
                " && mkdir -p ${{outputs.save_dir}}"
            ),
            inputs={
                "n_epoch": Input(type="integer"),
                "backbone": Input(type="string"),
                "learning_rate": Input(type="number"),
            },
            outputs={
                "save_dir": Output(type="uri_folder"),
            },
            environment="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
            compute=compute,
        )

    # -----------------------------------------------------------------------
    # @pipeline with typed function arguments
    # The SDK maps n_epoch: int → integer, backbone: str → string, etc.
    # -----------------------------------------------------------------------
    @pipeline(description="DSL pipeline demo from playground_azureml.py")  # type: ignore[no-matching-overload]
    def training_pipeline(
        n_epoch: int = 5,
        backbone: str = "effb0",
        learning_rate: float = 1e-4,
    ):
        train_step = _make_train_component()(
            n_epoch=n_epoch,
            backbone=backbone,
            learning_rate=learning_rate,
        )
        return {"save_dir": train_step.outputs.save_dir}

    # Instantiate the pipeline — typed kwargs flow straight through
    pipeline_job = training_pipeline(
        n_epoch=n_epoch,
        backbone=backbone,
        learning_rate=learning_rate,
    )
    pipeline_job.settings.default_compute = "cpu-cluster"

    typer.echo("\n" + "=" * 70)
    typer.echo("DSL @pipeline with typed parameters")
    typer.echo("=" * 70)
    typer.echo("\n  Pipeline inputs passed at instantiation time:")
    typer.echo(f"    n_epoch       = {n_epoch}  (int  → integer)")
    typer.echo(f"    backbone      = '{backbone}'  (str  → string)")
    typer.echo(f"    learning_rate = {learning_rate}  (float → number)")
    typer.echo()

    resolved_inputs = pipeline_job.inputs
    typer.echo("  Resolved pipeline_job.inputs (type → internal representation):")
    for name, inp in resolved_inputs.items():
        # PipelineInput wraps the scalar; extract _data or fall back to repr
        inner = getattr(inp, "_data", None)
        if inner is None:
            inner = getattr(inp, "_value", inp)
        value_repr = getattr(inner, "value", inner)
        typer.echo(f"    {name:<18} = {value_repr!r}  ({type(value_repr).__name__})")

    typer.echo()
    if dry_run:
        typer.echo("  [dry-run] Pipeline built successfully.  Pass --no-dry-run to submit.\n")
    else:
        # Retrieve connection options from settings when not supplied via CLI
        settings = AzureMlSettings()
        sub_id = settings.azure_subscription_id
        rg = settings.azure_resource_group
        ws_name = settings.azure_workspace_name

        if not (sub_id and rg and ws_name):
            typer.echo(
                "  [error] Connection settings incomplete for submission.\n"
                "          Set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE_NAME\n"
                "          in .env or pass them as environment variables.\n",
                err=True,
            )
            raise typer.Exit(code=1)

        ml_client = _get_ml_client(sub_id, rg, ws_name)
        submitted = ml_client.jobs.create_or_update(pipeline_job, experiment_name="playground-dsl-pipeline")
        typer.echo(f"  ✅ Submitted: {submitted.name}  status={submitted.status}\n")

    typer.echo("=" * 70 + "\n")


# ===========================================================================
# Entrypoint
# ===========================================================================

if __name__ == "__main__":
    load_dotenv(override=True, verbose=True)
    app()
