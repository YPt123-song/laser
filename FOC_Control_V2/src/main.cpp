/**
 * 基于 SimpleFOC 的双电机视觉伺服控制程序
 * 适配硬件：灯哥开源 FOC V3.0
 * * 更新日志：
 * - v7.2: 架构修正版 (Architecture Fix)
 * - 1. 修复 VisualPID 积分限幅 Bug (负 I 导致限幅失效)
 * - 2. PID 参数统一为正数 (Positive Params)
 * - 3. 在应用层 (updateVisualServoing) 处理方向 (Y轴使用 -=)
 * - 4. 增加详细的符号诊断打印
 */

#include <Arduino.h>
#include <SimpleFOC.h>
#include <math.h>

// ================== I2C 安全传感器封装 ==================
class SafeMagneticSensorI2C : public MagneticSensorI2C {
public:
  using MagneticSensorI2C::MagneticSensorI2C; 
  float last_angle = 0.0f;
  bool first = true;
  int consecutive_errors = 0;
  float max_step = 0.50f; 
  int max_consecutive_errors = 1; 

  float getSensorAngle() override {
    float a = MagneticSensorI2C::getSensorAngle();
    if (isnan(a) || isinf(a)) return last_angle;
    if (first) { first = false; last_angle = a; return a; }
    float diff = a - last_angle;
    if (diff > _PI)  diff -= _2PI;
    if (diff < -_PI) diff += _2PI;
    if (fabs(diff) > max_step) {
       consecutive_errors++;
       if (consecutive_errors > max_consecutive_errors) {
          consecutive_errors = 0; last_angle = a; return a;
       }
       return last_angle;
    }
    consecutive_errors = 0; last_angle = a; return a;
  }
};

// ================== 视觉 PID 类 (修复版) ==================
class VisualPID {
public:
    float P, I, D;
    float integral = 0.0f;
    float prev_error = 0.0f;
    float output_limit = 0.1f; 
    float integral_limit = 0.5f; 
    
    VisualPID(float p, float i, float d) : P(p), I(i), D(d) {}

    float compute(float error, float dt) {
        integral += error * dt;
        
        // 【Bug 修复】
        // 使用 fabs(I) 确保限幅边界正确，或者当 I=0 时避免除零
        // 这样即使 I 为负，限幅区间也是 [-min, +max]
        float limit = (fabs(I) > 1e-6) ? (integral_limit / fabs(I)) : 0.0f;
        integral = _constrain(integral, -limit, limit);

        float derivative = (error - prev_error) / dt;
        prev_error = error;
        
        float output = (P * error) + (I * integral) + (D * derivative);
        return _constrain(output, -output_limit, output_limit);
    }
};

// ================= 硬件配置 =================
int MOTOR0_PP = 7;  
int MOTOR1_PP = 7;  
#define SCREEN_CENTER_X 320.0f
#define SCREEN_CENTER_Y 240.0f

// 【参数归正】
// 全部使用正数，不再通过 PID 符号来猜方向
VisualPID pid_vis_x(0.0004f, 0.00001f, 0.0f); 
VisualPID pid_vis_y(0.0004f, 0.00001f, 0.0f);  

float target_angle_M1 = 0.0f; 
float target_angle_M2 = 0.0f; 

float zero_offset_M1 = 0.0f;
float zero_offset_M2 = 0.0f;

float visual_target_x = 0;
float visual_target_y = 0;
bool is_target_updated = false;
unsigned long last_visual_update = 0;

BLDCMotor motor1 = BLDCMotor(MOTOR0_PP); 
BLDCDriver3PWM driver1 = BLDCDriver3PWM(32, 33, 25, 12);

BLDCMotor motor2 = BLDCMotor(MOTOR1_PP);
BLDCDriver3PWM driver2 = BLDCDriver3PWM(26, 27, 14, 12);

SafeMagneticSensorI2C sensor1 = SafeMagneticSensorI2C(AS5600_I2C);
TwoWire I2Cone = TwoWire(0);

SafeMagneticSensorI2C sensor2 = SafeMagneticSensorI2C(AS5600_I2C);
TwoWire I2Ctwo = TwoWire(1);

String serial_buffer = "";
bool is_parsing = false;

void serialReceiveUserCommand();
void updateVisualServoing();

void setup() {
  Serial.begin(115200);
  
  Serial.println("\n\n>>> FIRMWARE V7.2 ARCHITECTURE FIX <<<\n");
  
  delay(2000); 

  pinMode(12, OUTPUT);
  digitalWrite(12, HIGH);
  delay(100);

  pinMode(19, INPUT_PULLUP);
  pinMode(18, INPUT_PULLUP);
  pinMode(23, INPUT_PULLUP);
  pinMode(5, INPUT_PULLUP);

  Serial.println("Init I2C (75k)...");

  I2Cone.begin(19, 18, 75000UL);   
  I2Ctwo.begin(23, 5, 75000UL);    
  I2Cone.setTimeOut(200); 
  I2Ctwo.setTimeOut(200);

  sensor1.init(&I2Cone);
  sensor2.init(&I2Ctwo);
  motor1.linkSensor(&sensor1);
  motor2.linkSensor(&sensor2);

  driver1.voltage_power_supply = 12.6;
  driver2.voltage_power_supply = 12.6;
  driver1.init();
  driver2.init();
  motor1.linkDriver(&driver1);
  motor2.linkDriver(&driver2);

  motor1.controller = MotionControlType::angle;
  motor2.controller = MotionControlType::angle;
  motor1.torque_controller = TorqueControlType::voltage;
  motor2.torque_controller = TorqueControlType::voltage;

  // --- M1 (底座) ---
  motor1.voltage_sensor_align = 4.0; 
  motor1.voltage_limit        = 6.0; 
  motor1.velocity_limit       = 20.0;
  motor1.PID_velocity.P       = 0.7;   
  motor1.PID_velocity.I       = 2.0;   
  motor1.PID_velocity.output_ramp = 10000.0;
  motor1.LPF_velocity.Tf      = 0.05;  
  motor1.P_angle.P            = 6.0;   

  // --- M2 (俯仰) ---
  motor2.voltage_sensor_align = 1.5; 
  motor2.voltage_limit        = 4.0;   
  motor2.velocity_limit       = 20.0;  
  motor2.PID_velocity.P       = 0.45;  
  motor2.PID_velocity.I       = 1.5;   
  motor2.PID_velocity.output_ramp = 10000.0;
  motor2.LPF_velocity.Tf      = 0.05;  
  motor2.P_angle.P            = 4.5;   

  Serial.println("Aligning M1...");
  motor1.init();
  motor1.initFOC(); 

  Serial.println("Aligning M2...");
  motor2.init();
  motor2.initFOC(); 

  motor1.loopFOC(); 
  motor2.loopFOC();
  zero_offset_M1 = motor1.shaft_angle;
  zero_offset_M2 = motor2.shaft_angle;

  target_angle_M1 = 0.0f;
  target_angle_M2 = 0.0f;

  Serial.printf("Ready. Home Set: M1=%.2f, M2=%.2f\n", zero_offset_M1, zero_offset_M2);
}

void loop() {
  motor1.loopFOC();
  motor2.loopFOC();

  motor1.move(target_angle_M1 + zero_offset_M1);
  motor2.move(target_angle_M2 + zero_offset_M2);

  serialReceiveUserCommand();
  updateVisualServoing();

  static unsigned long last_print_time = 0;
  if (millis() - last_print_time > 500) { // 降低频率，避免刷屏 
    last_print_time = millis();
    // 打印当前相对角度
    Serial.printf("M1:%.2f M2:%.2f\n", 
                  motor1.shaft_angle - zero_offset_M1, 
                  motor2.shaft_angle - zero_offset_M2);
  }
}

void serialReceiveUserCommand() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '$') { serial_buffer = ""; is_parsing = true; continue; }
    if (is_parsing) {
      if (inChar == '&') {
        is_parsing = false;
        int commaPos = serial_buffer.indexOf(',');
        if (commaPos != -1) {
           String type = serial_buffer.substring(0, commaPos);
           String data = serial_buffer.substring(commaPos + 1);
           if (type == "target") {
              int slashPos = data.indexOf('/');
              if (slashPos != -1) {
                  visual_target_x = data.substring(0, slashPos).toFloat();
                  visual_target_y = data.substring(slashPos + 1).toFloat();
                  is_target_updated = true;
              }
           }
        }
      }
      else if (inChar == '*') { 
        is_parsing = false;
        int commaPos = serial_buffer.indexOf(',');
        if (commaPos != -1) {
            String cmd_type = serial_buffer.substring(0, commaPos);
            String cmd_value = serial_buffer.substring(commaPos + 1);
            if (cmd_type == "VPID_X") {
                int c1 = cmd_value.indexOf('/');
                int c2 = cmd_value.indexOf('/', c1+1);
                if (c1!=-1 && c2!=-1) {
                    pid_vis_x.P = cmd_value.substring(0, c1).toFloat();
                    pid_vis_x.I = cmd_value.substring(c1+1, c2).toFloat();
                    pid_vis_x.D = cmd_value.substring(c2+1).toFloat();
                }
            }
        }
      }
      else if (inChar == '#') { 
        is_parsing = false;
        int commaPos = serial_buffer.indexOf(',');
        if (commaPos != -1) {
            String cmd_type = serial_buffer.substring(0, commaPos);
            String cmd_value = serial_buffer.substring(commaPos + 1);
            if (cmd_type == "Angle_0") target_angle_M1 = cmd_value.toFloat();
            else if (cmd_type == "Angle_1") target_angle_M2 = cmd_value.toFloat();
        }
      }
      else { serial_buffer += inChar; }
    }
  }
}

void updateVisualServoing() {
  if (is_target_updated) {
    unsigned long now = millis();
    float dt = (now - last_visual_update) / 1000.0f;
    last_visual_update = now;
    if (dt <= 0 || dt > 0.5f) dt = 0.05f; 

    float error_x = visual_target_x - SCREEN_CENTER_X;
    float error_y = visual_target_y - SCREEN_CENTER_Y;

    if (fabs(error_x) < 10) error_x = 0;
    if (fabs(error_y) < 10) error_y = 0;

    // 计算 PID 输出 (角度增量)，使用正数 PID 参数
    float delta_x = pid_vis_x.compute(error_x, dt);
    float delta_y = pid_vis_y.compute(error_y, dt);

    // 【调试诊断】
    // 打印误差和计算出的增量，观察方向是否符合逻辑
    // 例子：如果物体在画面上方 (y < 240)，error_y 为负。
    // 如果 delta_y 为负，M2 角度减小。
    // Serial.printf("Y_Err:%.0f -> dY:%.4f | M2_Tgt:%.2f\n", error_y, delta_y, target_angle_M2);

    // 【方向控制层】
    // 在这里决定正负反馈，而不是改 PID 符号
    target_angle_M1 += delta_x; // X轴：保持 += (假设左右正常)
    
    // Y轴：改为 -= 
    // 逻辑：屏幕 Y 增大(向下)，Error > 0。此时我们需要电机向下转。
    // 如果电机正转是向上，那么这里需要减去正的 delta_y。
    target_angle_M2 -= delta_y; 

    target_angle_M1 = _constrain(target_angle_M1, -3.0f, 3.0f);
    target_angle_M2 = _constrain(target_angle_M2, -3.0f, 3.0f);

    is_target_updated = false;
  }
}