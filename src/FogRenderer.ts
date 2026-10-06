const VERTEX_SHADER = `#version 300 es
in vec2 a_position;
out vec2 v_uv;
void main() {
  v_uv = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;
uniform sampler2D u_video;
uniform sampler2D u_mask;
uniform vec2 u_resolution;
uniform float u_time;
in vec2 v_uv;
out vec4 outColor;

float hash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash21(i), hash21(i + vec2(1, 0)), f.x),
             mix(hash21(i + vec2(0, 1)), hash21(i + vec2(1)), f.x), f.y);
}

float fbm(vec2 p) {
  float value = 0.0;
  float amplitude = 0.52;
  for (int i = 0; i < 4; i++) {
    value += noise(p) * amplitude;
    p = p * 2.03 + 17.17;
    amplitude *= 0.48;
  }
  return value;
}

vec4 sampleVideo(vec2 uv) {
  vec2 mirrored = vec2(1.0 - clamp(uv.x, 0.0, 1.0), clamp(uv.y, 0.0, 1.0));
  return texture(u_video, mirrored);
}

vec3 softVideo(vec2 uv, float radius) {
  vec2 px = radius / u_resolution;
  vec3 color = sampleVideo(uv).rgb * 0.24;
  color += sampleVideo(uv + vec2(px.x, 0)).rgb * 0.13;
  color += sampleVideo(uv - vec2(px.x, 0)).rgb * 0.13;
  color += sampleVideo(uv + vec2(0, px.y)).rgb * 0.13;
  color += sampleVideo(uv - vec2(0, px.y)).rgb * 0.13;
  color += sampleVideo(uv + px).rgb * 0.06;
  color += sampleVideo(uv - px).rgb * 0.06;
  color += sampleVideo(uv + vec2(px.x, -px.y)).rgb * 0.06;
  color += sampleVideo(uv + vec2(-px.x, px.y)).rgb * 0.06;
  return color;
}

void main() {
  vec2 uv = v_uv;
  float cleared = texture(u_mask, uv).r;
  float fog = 1.0 - cleared;

  float n1 = fbm(uv * vec2(4.2, 3.1) + vec2(u_time * 0.012, -u_time * 0.008));
  float n2 = fbm(uv * 13.0 - vec2(u_time * 0.006));
  vec2 distortion = vec2(
    fbm(uv * 8.0 + vec2(0.0, u_time * 0.01)) - 0.5,
    fbm(uv * 8.0 + vec2(8.4, -u_time * 0.008)) - 0.5
  );

  vec2 refractedUv = uv + distortion * 0.003 * fog;
  vec3 clearVideo = sampleVideo(refractedUv).rgb;
  vec3 blurredVideo = softVideo(refractedUv, 8.0 + fog * 14.0);

  vec3 coolFog = mix(vec3(0.72, 0.79, 0.80), vec3(0.91, 0.94, 0.94), n1);
  coolFog += (n2 - 0.5) * 0.028;
  vec3 fogged = mix(blurredVideo, coolFog, 0.72 + n1 * 0.12);

  vec2 px = 2.0 / u_resolution;
  float nearMask = max(max(texture(u_mask, uv + vec2(px.x, 0)).r, texture(u_mask, uv - vec2(px.x, 0)).r),
                       max(texture(u_mask, uv + vec2(0, px.y)).r, texture(u_mask, uv - vec2(0, px.y)).r));
  float edge = clamp(nearMask - cleared, 0.0, 1.0);

  vec3 color = mix(fogged, clearVideo, smoothstep(0.02, 0.82, cleared));
  color += vec3(0.72, 0.92, 0.94) * edge * 0.14;

  float vignette = smoothstep(0.92, 0.28, distance(uv, vec2(0.5)));
  color *= mix(0.84, 1.0, vignette);
  outColor = vec4(color, 1.0);
}`;

export class FogRenderer {
  private readonly gl: WebGL2RenderingContext;
  private readonly program: WebGLProgram;
  private readonly videoTexture: WebGLTexture;
  private readonly maskTexture: WebGLTexture;
  private readonly resolutionLocation: WebGLUniformLocation;
  private readonly timeLocation: WebGLUniformLocation;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly video: HTMLVideoElement
  ) {
    const gl = canvas.getContext('webgl2', { alpha: false, antialias: false });
    if (!gl) throw new Error('WebGL2 is required for the realistic fog renderer.');
    this.gl = gl;
    this.program = this.createProgram(VERTEX_SHADER, FRAGMENT_SHADER);
    this.resolutionLocation = this.uniform('u_resolution');
    this.timeLocation = this.uniform('u_time');

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const position = gl.getAttribLocation(this.program, 'a_position');
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);

    this.videoTexture = this.createTexture(0);
    this.maskTexture = this.createTexture(1);
    gl.useProgram(this.program);
    gl.uniform1i(this.uniform('u_video'), 0);
    gl.uniform1i(this.uniform('u_mask'), 1);
  }

  resize(width: number, height: number, dpr: number): void {
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(height * dpr);
    this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
  }

  render(mask: HTMLCanvasElement, timeMs: number): void {
    const gl = this.gl;
    gl.useProgram(this.program);
    if (this.video.readyState >= 2) {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.videoTexture);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.video);
    }
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.maskTexture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, mask);
    gl.uniform2f(this.resolutionLocation, this.canvas.width, this.canvas.height);
    gl.uniform1f(this.timeLocation, timeMs / 1000);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  private createTexture(unit: number): WebGLTexture {
    const texture = this.gl.createTexture();
    if (!texture) throw new Error('Unable to create WebGL texture.');
    this.gl.activeTexture(this.gl.TEXTURE0 + unit);
    this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
    this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, 1, 1, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
    return texture;
  }

  private createProgram(vertexSource: string, fragmentSource: string): WebGLProgram {
    const program = this.gl.createProgram();
    if (!program) throw new Error('Unable to create WebGL program.');
    this.gl.attachShader(program, this.createShader(this.gl.VERTEX_SHADER, vertexSource));
    this.gl.attachShader(program, this.createShader(this.gl.FRAGMENT_SHADER, fragmentSource));
    this.gl.linkProgram(program);
    if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
      throw new Error(this.gl.getProgramInfoLog(program) || 'Unable to link fog shader.');
    }
    return program;
  }

  private createShader(type: number, source: string): WebGLShader {
    const shader = this.gl.createShader(type);
    if (!shader) throw new Error('Unable to create WebGL shader.');
    this.gl.shaderSource(shader, source);
    this.gl.compileShader(shader);
    if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
      throw new Error(this.gl.getShaderInfoLog(shader) || 'Unable to compile fog shader.');
    }
    return shader;
  }

  private uniform(name: string): WebGLUniformLocation {
    const location = this.gl.getUniformLocation(this.program, name);
    if (!location) throw new Error(`Missing shader uniform: ${name}`);
    return location;
  }
}
