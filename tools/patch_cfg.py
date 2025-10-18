#!/usr/bin/env python
import argparse, yaml, os

def main():
    parser = argparse.ArgumentParser(description="Patch config file with command-line settings")
    parser.add_argument("--cfg", default="configs/reid.yaml", help="Config file path")
    parser.add_argument("--set", action="append", help="Set key=value pairs")
    args = parser.parse_args()
    
    # Load existing config
    with open(args.cfg, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Apply patches
    if args.set:
        for setting in args.set:
            if "=" not in setting:
                print(f"Warning: Invalid setting format '{setting}', skipping")
                continue
            key, value = setting.split("=", 1)
            keys = key.split(".")
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            # Convert value types
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            elif value.isdigit():
                value = int(value)
            elif value.replace(".", "").isdigit():
                value = float(value)
            elif value == "null":
                value = None
            current[keys[-1]] = value
            print(f"Set {key} = {value}")
    
    # Save updated config
    with open(args.cfg, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
    print(f"Updated config saved to {args.cfg}")

if __name__ == "__main__":
    main()