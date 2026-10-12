import { api, pickFile, pickSave, VIDEO_EXTENSIONS } from "../lib/ipc";
import { basename, dirname, stem } from "../lib/format";
import { openSource, setPipeline, toast, useStore } from "../state/store";

const FILTER = [{ name: "ClipFX Enhance project", extensions: ["cfxproj"] }];

export async function openProjectFile(path?: string) {
  const p = path ?? (await pickFile("Open project", FILTER))[0];
  if (!p) return;
  const r = await api.projectOpen(p);
  let src = r.project.source.path;
  if (!r.source_found) {
    if (r.suggested_source) {
      src = r.suggested_source;
      toast("info", `The source video moved; using ${src}.`);
    } else {
      toast("warning", `The source video ${basename(src)} was not found. Choose where it is now.`);
      const [np] = await pickFile(`Locate ${basename(src)}`, [{ name: "Video", extensions: VIDEO_EXTENSIONS }]);
      if (!np) return;
      src = np;
    }
  } else if (r.source_changed) {
    toast("warning", "The source video changed since this project was saved. Check the preview before exporting.");
  }
  await openSource(src);
  const pr = r.project;
  setPipeline(pr.pipeline, false);
  useStore.setState({
    output: pr.output, range: pr.range, engine: pr.engine, exportFolder: pr.export.folder, exportFileName: pr.export.file_name,
    projectPath: p, projectName: pr.name, dirty: false, history: { past: [], future: [] }, dialog: null,
  });
  if (r.migrated_from !== null) toast("info", `Project upgraded from format ${r.migrated_from}. A backup was kept${r.backup_path ? ` at ${r.backup_path}` : ""}.`);
  if (r.missing_models.length) toast("warning", `This project uses models that are not installed: ${r.missing_models.join(", ")}.`, { label: "Models", run: () => useStore.setState({ dialog: "models" }) });
}

export async function saveProject(saveAs = false) {
  const s = useStore.getState();
  if (!s.sourcePath || !s.output) return;
  let path = s.projectPath;
  if (!path || saveAs) {
    path = await pickSave("Save project", `${dirname(s.sourcePath)}/${stem(s.sourcePath)}.cfxproj`, FILTER);
    if (!path) return;
    if (!path.toLowerCase().endsWith(".cfxproj")) path += ".cfxproj";
  }
  const name = s.projectName ?? stem(path);
  const base = await api.newProject(s.sourcePath, name);
  const project = {
    ...base,
    pipeline: s.pipeline, output: s.output, range: s.range, engine: s.engine,
    export: { folder: s.exportFolder, file_name: s.exportFileName },
  };
  await api.projectSave(path, project);
  useStore.setState({ projectPath: path, projectName: name, dirty: false });
  toast("success", `Saved ${basename(path)}.`);
}
