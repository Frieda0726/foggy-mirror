# Foggy Mirror / 雾面镜

一个完全在浏览器本地运行的互动雾面镜：抬手后自动识别食指指尖，短暂防误触缓冲后直接移动食指写字；张嘴哈气会把嘴前方的雾气吹开，随后镜面会缓慢重新凝结。

## 功能

- MediaPipe Hand Landmarker 直接追踪食指指尖；连续移动优先保持完整笔画，字母间停顿会智能抬笔
- 首次识别食指后自动学习手掌尺度与移动速度，并据此调整笔宽和分笔阈值
- 本地分析闭合轨迹；画出爱心后会自动给出识别反馈
- MediaPipe Face Landmarker 先校准闭嘴基线，再通过明显、持续的张嘴动作识别哈气，减少误触发
- Canvas 实时雾化、局部擦除与嘴部附近的小范围哈气晕染
- WebGL2 摄像头纹理与动态磨砂冷凝噪声
- 擦除边缘辉光以及随时间自然恢复的雾气
- 鼠标或触控书写回退
- 摄像头画面和识别均留在本机，不上传、不保存

> 普通摄像头不能直接检测气流，因此“哈气”功能根据噘嘴动作及其持续时间进行视觉推断。

## 本地运行

```bash
npm install
npm run dev
```

打开 `http://127.0.0.1:2502`，允许摄像头权限。

## 来源与署名

本项目从 [quiet-node/gesture-lab](https://github.com/quiet-node/gesture-lab) 的 **Foggy Mirror** 实验独立演化而来。原项目由 [quiet-node](https://github.com/quiet-node) 创建；本项目保留原 MIT 版权声明，并由 [Frieda0726](https://github.com/Frieda0726) 继续维护和扩展。

视觉渲染研究参考了 [Binix Rainy Glass Simulator](https://github.com/CMbin2333/Binix-Rainy-Glass-Simulator)、[Amado](https://github.com/Oililyuk/amado)、[canvas-effects](https://github.com/TokyoDanInJapan/canvas-effects) 与 [WebLG](https://github.com/KhanUnix/WebLG)。本项目的着色器与交互实现为独立编写。

## License

[MIT](LICENSE)
