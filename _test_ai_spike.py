"""AI 背景图 API spike test。

用法：
    export OPENAI_API_KEY="sk-..."
    export OPENAI_BASE_URL="https://your-proxy.com/v1"  # 可选

    python3 _test_ai_spike.py
    python3 _test_ai_spike.py --model gpt-image-2
    python3 _test_ai_spike.py --model dall-e-3
"""
import os
import sys
import time
import argparse

from core.ai_background import AIConfig, generate_backgrounds, test_connection

DEFAULT_PROMPTS = {
    "1:1": "A realistic photograph of a laptop computer on a wooden desk, screen showing solid black, warm ambient lighting",
    "4:3": "A realistic photograph of a desktop monitor on an office desk, screen showing solid black, natural light from window",
    "16:9": "A realistic photograph of a Chinese elementary school classroom with a smart board, screen showing solid black, blackboard visible",
}

def main():
    parser = argparse.ArgumentParser(description="AI 背景图 API spike test")
    parser.add_argument("--model", default=None, help="指定模型名（默认尝试 gpt-image-2 + dall-e-3）")
    parser.add_argument("--ratios", nargs="+", default=["1:1"], help="测试比例（默认只测 1:1）")
    parser.add_argument("--save-dir", default=None, help="保存生成的图片到此目录")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if not api_key:
        print("未设置 OPENAI_API_KEY 环境变量")
        sys.exit(1)

    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"Base URL: {base_url}")
    print()

    models = [args.model] if args.model else ["gpt-image-2", "dall-e-3"]

    results = []
    for model in models:
        print(f"--- 测试模型: {model} ---")
        config = AIConfig(api_key=api_key, base_url=base_url, model=model)
        for ratio in args.ratios:
            print(f"  比例 {ratio}: ", end="", flush=True)
            t0 = time.time()
            try:
                imgs = generate_backgrounds(
                    config,
                    DEFAULT_PROMPTS.get(ratio, DEFAULT_PROMPTS["1:1"]),
                    n=1, aspect_ratio=ratio,
                )
                elapsed = time.time() - t0
                w, h = imgs[0].size
                print(f"成功（{elapsed:.1f}s, {w}x{h}）")
                results.append((model, ratio, "ok", elapsed, f"{w}x{h}"))

                if args.save_dir:
                    os.makedirs(args.save_dir, exist_ok=True)
                    out = os.path.join(args.save_dir, f"{model}_{ratio.replace(':', 'x')}.png")
                    imgs[0].save(out)
                    print(f"     已保存: {out}")
            except Exception as e:
                elapsed = time.time() - t0
                err_type = type(e).__name__
                print(f"失败（{elapsed:.1f}s）: {err_type}: {e}")
                results.append((model, ratio, "fail", elapsed, str(e)))
        print()

    print("--- 总结 ---")
    ok_combos = [r for r in results if r[2] == "ok"]
    if ok_combos:
        print(f"{len(ok_combos)} 个组合可用：")
        for model, ratio, _, elapsed, info in ok_combos:
            print(f"  - {model} + {ratio} （{elapsed:.1f}s, {info}）")
        print()
        print("推荐用第一个组合作为默认值。把这个反馈给 Echo，它会调整 AIConfig 默认值。")
    else:
        print("所有组合都失败了。把上面的具体错误信息反馈给 Echo 排查。")
        sys.exit(1)


if __name__ == "__main__":
    main()
