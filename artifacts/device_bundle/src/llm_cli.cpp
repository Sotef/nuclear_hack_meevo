// llm_cli.cpp — minimal on-device LLM CLI for ONNX Runtime GenAI (Android arm64).
//
// Loads an ORT GenAI model directory (genai_config.json + model.onnx + tokenizer),
// feeds a prompt, runs greedy decoding for up to N tokens and prints timing.
//
// Build (NDK r27c, aarch64-linux-android24):
//   clang++ --target=aarch64-linux-android24 -std=c++17 -O2 -fPIE -pie \
//     -I<ort_genai headers> llm_cli.cpp -L<libdir> -lonnxruntime-genai -lmat \
//     -llog -ldl -lm -o llm_cli
//
// Runtime: libonnxruntime-genai.so and libmat.so must be next to the binary or
// on LD_LIBRARY_PATH.
//
// Usage:
//   llm_cli --model <dir> --prompt "text" [--max-new-tokens 64]
//   llm_cli --model <dir> --chatml --system "..." --user "..." [--max-new-tokens 64]
//   llm_cli --model <dir> --prompt-file prompt.txt [--max-new-tokens 64]

#include "ort_genai.h"

#include <chrono>
#include <cstdio>
#include <exception>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Force PT_TLS alignment >= 64 (bionic on arm64 rejects underaligned TLS
// segments produced by lld when linking against libc++_shared.so).
namespace { alignas(64) thread_local char g_tls_align_pad; }

static void term_handler() {
  try {
    std::rethrow_exception(std::current_exception());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "TERMINATE: %s\n", e.what());
  } catch (...) {
    std::fprintf(stderr, "TERMINATE: unknown exception\n");
  }
  std::abort();
}

static void usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s --model <dir> (--prompt <text> | --prompt-file <path> | "
               "--chatml --system <s> --user <u>) [--max-new-tokens N]\n",
               argv0);
}

static std::string read_file(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    std::fprintf(stderr, "ERROR: cannot open %s\n", path.c_str());
    std::exit(2);
  }
  std::string out;
  char buf[65536];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
  std::fclose(f);
  return out;
}

// MiniCPM5 ChatML with enable_thinking=false (see chat_template.jinja).
static std::string build_chatml(const std::string& system, const std::string& user) {
  std::string p;
  p += "<s>";
  p += "<|im_start|>system\n" + system + "<|im_end|>\n";
  p += "<|im_start|>user\n" + user + "<|im_end|>\n";
  p += "<|im_start|>assistant\n";
  p += " thinking\n\n</think>\n\n";
  return p;
}

int main(int argc, char** argv) {
  std::string model_dir;
  std::string prompt;
  std::string prompt_file;
  std::string system;
  std::string user;
  bool chatml = false;
  int max_new = 64;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "ERROR: %s requires a value\n", name);
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--model") model_dir = next("--model");
    else if (a == "--prompt") prompt = next("--prompt");
    else if (a == "--prompt-file") prompt_file = next("--prompt-file");
    else if (a == "--system") system = next("--system");
    else if (a == "--user") user = next("--user");
    else if (a == "--chatml") chatml = true;
    else if (a == "--max-new-tokens") max_new = std::atoi(next("--max-new-tokens").c_str());
    else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
    else {
      std::fprintf(stderr, "ERROR: unknown argument %s\n", a.c_str());
      usage(argv[0]);
      return 2;
    }
  }

  if (model_dir.empty()) { usage(argv[0]); return 2; }
  if (!prompt_file.empty()) prompt = read_file(prompt_file);
  if (chatml) prompt = build_chatml(system, user);
  if (prompt.empty()) {
    std::fprintf(stderr, "ERROR: empty prompt (use --prompt/--prompt-file/--chatml)\n");
    return 2;
  }
  if (max_new <= 0) max_new = 64;
  g_tls_align_pad = 0;
  std::set_terminate(term_handler);

  try {
  using clock = std::chrono::steady_clock;
  auto t_load0 = clock::now();
  auto model = OgaModel::Create(model_dir.c_str());
  auto tokenizer = OgaTokenizer::Create(*model);
  auto t_load1 = clock::now();

  auto sequences = OgaSequences::Create();
  tokenizer->Encode(prompt.c_str(), *sequences);
  const size_t prompt_len = sequences->SequenceCount(0);

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", static_cast<double>(prompt_len + max_new));
  params->SetSearchOption("min_length", static_cast<double>(prompt_len));
  params->SetSearchOptionBool("do_sample", false);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokenSequences(*sequences);

  auto t_gen0 = clock::now();
  size_t generated = 0;
  double ttft_ms = -1.0;
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
    ++generated;
    if (generated == 1) {
      ttft_ms = std::chrono::duration<double, std::milli>(clock::now() - t_gen0).count();
    }
  }
  auto t_gen1 = clock::now();

  const int32_t* seq = generator->GetSequenceData(0);
  const size_t total = generator->GetSequenceCount(0);
  const size_t new_count = total > prompt_len ? total - prompt_len : 0;
  auto out = tokenizer->Decode(seq + prompt_len, new_count);

  const double load_ms = std::chrono::duration<double, std::milli>(t_load1 - t_load0).count();
  const double gen_ms = std::chrono::duration<double, std::milli>(t_gen1 - t_gen0).count();
  const double decode_tps = gen_ms > 0 ? (generated * 1000.0 / gen_ms) : 0.0;

  std::printf("RESULT:%s\n", static_cast<const char*>(out));
  std::printf(
      "STATS: prompt_tokens=%zu generated_tokens=%zu load_ms=%.1f ttft_ms=%.1f "
      "gen_ms=%.1f decode_tok_s=%.2f\n",
      prompt_len, generated, load_ms, ttft_ms, gen_ms, decode_tps);
  std::fflush(stdout);
  return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
    return 3;
  }
}
