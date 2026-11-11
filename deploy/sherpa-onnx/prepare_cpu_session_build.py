"""Prepare an opt-in NeMo CPU session build from pinned sherpa-onnx 1.13.4.

Run against a fresh extraction of upstream's source archive. No live runtime
files are edited. The generated patch and source hashes make the build auditable.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path


def replace(text, old, new, count=1):
    assert text.count(old) == count, (old[:100], text.count(old), count)
    return text.replace(old, new)


def prepare(root: Path, output: Path):
    root = root.resolve()
    assert 'set(SHERPA_ONNX_VERSION "1.13.4")' in (root / "CMakeLists.txt").read_text()
    paths = ["sherpa-onnx/csrc/session.h", "sherpa-onnx/csrc/session.cc",
             "sherpa-onnx/csrc/offline-transducer-nemo-model.cc", "setup.py"]
    original = {name: (root / name).read_text() for name in paths}
    changed = dict(original)
    name = paths[0]
    changed[name] = replace(changed[name],
        'const ProviderConfig *provider_config = nullptr);',
        'const ProviderConfig *provider_config = nullptr,\n'
        '    const std::string &component = "");')
    changed[name] = replace(changed[name],
        '}  // namespace sherpa_onnx',
        'template <typename T>\n'
        'Ort::SessionOptions GetComponentSessionOptions(\n'
        '    const T &config, const std::string &component) {\n'
        '  return GetSessionOptionsImpl(config.num_threads, config.provider,\n'
        '                               nullptr, component);\n'
        '}\n\n'
        '}  // namespace sherpa_onnx')
    name = paths[1]
    changed[name] = replace(changed[name],
        'const ProviderConfig *provider_config /*= nullptr*/) {',
        'const ProviderConfig *provider_config /*= nullptr*/,\n'
        '    const std::string &component /*= ""*/) {')
    changed[name] = replace(changed[name],
        '  Ort::SessionOptions sess_opts;\n  sess_opts.SetIntraOpNumThreads(num_threads);',
        '''  // CPU component overrides are opt-in; unchanged configs retain upstream defaults.
  if (p == Provider::kCPU && !component.empty()) {
    const auto key = component + "NumThreads";
    auto it = config.find(key);
    if (it != config.end()) {
      num_threads = ToIntOrDefault(it->second, 0);
      if (num_threads < 1) {
        SHERPA_ONNX_LOGE("Invalid %s: %s", key.c_str(), it->second.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
    }
  }
  Ort::SessionOptions sess_opts;
  sess_opts.SetIntraOpNumThreads(num_threads);''')
    changed[name] = replace(changed[name],
        '\n  std::vector<std::string> available_providers = Ort::GetAvailableProviders();',
        '''
  if (p == Provider::kCPU) {
    auto inter = config.find("InterOpNumThreads");
    if (inter != config.end()) {
      const auto count = ToIntOrDefault(inter->second, 0);
      if (count < 1) {
        SHERPA_ONNX_LOGE("Invalid InterOpNumThreads: %s", inter->second.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      sess_opts.SetInterOpNumThreads(count);
    }
    auto spinning = config.find("AllowSpinning");
    if (spinning != config.end()) {
      if (spinning->second != "0" && spinning->second != "1") {
        SHERPA_ONNX_LOGE("AllowSpinning must be 0 or 1");
        SHERPA_ONNX_EXIT(-1);
      }
      sess_opts.AddConfigEntry("session.intra_op.allow_spinning", spinning->second.c_str());
      sess_opts.AddConfigEntry("session.inter_op.allow_spinning", spinning->second.c_str());
    }
  }

  std::vector<std::string> available_providers = Ort::GetAvailableProviders();''')
    changed[name] = replace(changed[name],
        'sess_opts.EnableProfiling(SHERPA_ONNX_TO_ORT_PATH(config["ProfilingFilePrefix"]));',
        'const auto prefix = config["ProfilingFilePrefix"] +\n'
        '                        (component.empty() ? "" : "-" + component);\n'
        '    sess_opts.EnableProfiling(SHERPA_ONNX_TO_ORT_PATH(prefix));')
    name = paths[2]
    changed[name] = replace(changed[name],
        '        sess_opts_(GetSessionOptions(config)),',
        '        encoder_opts_(GetComponentSessionOptions(config, "Encoder")),\n'
        '        decoder_opts_(GetComponentSessionOptions(config, "Decoder")),\n'
        '        joiner_opts_(GetComponentSessionOptions(config, "Joiner")),', count=2)
    for component in ("encoder", "decoder", "joiner"):
        changed[name] = replace(changed[name],
            f'config.transducer.{component}_filename),\n        sess_opts_);',
            f'config.transducer.{component}_filename),\n        {component}_opts_);')
        changed[name] = replace(changed[name],
            f'{component}_sess_ = std::make_unique<Ort::Session>(\n'
            '          env_, model_data, model_data_length, sess_opts_);',
            f'{component}_sess_ = std::make_unique<Ort::Session>(\n'
            f'          env_, model_data, model_data_length, {component}_opts_);')
    changed[name] = replace(changed[name], '  Ort::SessionOptions sess_opts_;',
        '  Ort::SessionOptions encoder_opts_;\n'
        '  Ort::SessionOptions decoder_opts_;\n'
        '  Ort::SessionOptions joiner_opts_;')
    changed[paths[3]] = replace(changed[paths[3]],
        '    return latest_version\n', '    return latest_version + "+orukeet.cpu1"\n')
    output.mkdir(parents=True, exist_ok=True)
    patch = "".join("".join(difflib.unified_diff(original[n].splitlines(True),
        changed[n].splitlines(True), fromfile="a/"+n, tofile="b/"+n)) for n in paths)
    (output / "cpu-sessions.patch").write_text(patch)
    hashes = {n: {"before": hashlib.sha256(original[n].encode()).hexdigest(),
                  "after": hashlib.sha256(changed[n].encode()).hexdigest()} for n in paths}
    (output / "source-changes.json").write_text(json.dumps(hashes, indent=2))
    for name in paths:
        (root / name).write_text(changed[name])
    print(json.dumps({"source": str(root), "patch": str(output / "cpu-sessions.patch"),
                      "changed_files": paths}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)
