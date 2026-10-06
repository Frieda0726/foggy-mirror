# Foggy Mirror / 雾面镜

一个完全在浏览器本地运行的互动雾面镜：抬手后自动识别食指指尖，短暂防误触缓冲后直接移动食指写字；张嘴哈气会把嘴前方的雾气吹开，随后镜面会缓慢重新凝结。

## 功能

- MediaPipe Hand Landmarker 直接追踪食指指尖，并用短暂时间缓冲避免抬手瞬间留下划痕
- MediaPipe Face Landmarker `mouthPucker` 表情分数识别哈气嘴形
- Canvas 实时雾化、局部擦除与局部重新起雾
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
