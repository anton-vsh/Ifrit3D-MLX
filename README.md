# Hunyuan3D on M{PS, LX}
The goal of this project is to port every Hunyuan3D 2.x model to run on Apple Silicon, and eventually, MLX.

# Current Status
| Model | Type | MPS | MLX | MLX HF |
| - | - | - | - | - |
| hunyuan3d-dit-v2-mini | 🧱 | ✅ | ❌ | |
| hunyuan3d-dit-v2-mini-turbo | 🧱 | ✅ | ❌ | |
| hunyuan3d-dit-v2-0 | 🧱 | ✅ | ❌ | |
| hunyuan3d-dit-v2-0-turbo | 🧱 | ✅ | ❌ | |
| hunyuan3d-dit-v2-1 | 🧱 | ✅ | ❌ | |
| hunyuan3d-dit-v2-mv | 🧱 | ✅ | ❌ | |
| hunyuan3d-dit-v2-mv-turbo | 🧱 | ✅ | ❌ | |
| hunyuan3d-paint-v2-0 | 🎨 | ✅ | ✅ | |
| hunyuan3d-paint-v2-0-turbo | 🎨 | ✅ | ✅ | |
| hunyuan3d-paintpbr-v2-1 | 🎨 | ✅ | ✅ | |

## 2.0 vs 2.0-turbo paint (structure)
`hunyuan3d-paint-v2-0` and `hunyuan3d-paint-v2-0-turbo` use the same core UNet tensor structure in practice (same key dimensions/config-level shape), which is why the same MLX conversion profile (`paint-2.0`) works for both.

The main runtime difference is mode/scheduler behavior: turbo is run through the turbo path (`hunyuanpaint-turbo`, LCM + turbo mode), while non-turbo uses the standard paint path.
