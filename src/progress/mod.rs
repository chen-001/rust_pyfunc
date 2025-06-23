use std::time::{Instant, Duration};
use std::sync::{Arc, Mutex};

/// 进度跟踪器
#[derive(Debug)]
pub struct ProgressTracker {
    start_time: Option<Instant>,
    total_tasks: usize,
    completed_tasks: usize,
    last_update_time: Option<Instant>,
    last_completed: usize,
}

impl ProgressTracker {
    pub fn new() -> Self {
        Self {
            start_time: None,
            total_tasks: 0,
            completed_tasks: 0,
            last_update_time: None,
            last_completed: 0,
        }
    }

    /// 使用已有结果开始跟踪
    pub fn start_with_existing(&mut self, total_tasks: usize, existing_tasks: usize, start_time: Instant) {
        self.start_time = Some(start_time);
        self.total_tasks = total_tasks;
        self.completed_tasks = existing_tasks;
        self.last_update_time = Some(Instant::now());
        self.last_completed = existing_tasks;
    }

    /// 开始跟踪
    pub fn start(&mut self, total_tasks: usize) {
        self.start_time = Some(Instant::now());
        self.total_tasks = total_tasks;
        self.completed_tasks = 0;
        self.last_update_time = Some(Instant::now());
        self.last_completed = 0;
    }

    /// 更新进度
    pub fn update(&mut self, increment: usize) {
        self.completed_tasks += increment;
        
        // 可以在这里添加进度显示逻辑
        if self.completed_tasks % 1000 == 0 || self.completed_tasks == self.total_tasks {
            self.print_progress();
        }
    }

    pub fn add_completed(&mut self, count: usize) {
        self.completed_tasks += count;
    }

    pub fn get_completed(&self) -> usize {
        self.completed_tasks
    }
    
    pub fn get_rate(&self) -> f64 {
        if let Some(start) = self.start_time {
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed > 0.0 {
                return self.completed_tasks as f64 / elapsed;
            }
        }
        0.0
    }
    
    pub fn get_progress_info(&self) -> (f64, f64) {
        if let Some(start) = self.start_time {
            let elapsed = start.elapsed().as_secs_f64();
            let remaining = if self.total_tasks > self.completed_tasks {
                let rate = self.get_rate();
                if rate > 0.0 {
                    (self.total_tasks - self.completed_tasks) as f64 / rate
                } else {
                    f64::INFINITY
                }
            } else {
                0.0
            };
            (remaining, elapsed)
        } else {
            (f64::INFINITY, 0.0)
        }
    }

    /// 获取当前进度信息
    pub fn get_progress(&self) -> ProgressInfo {
        let elapsed = self.start_time
            .map(|start| start.elapsed())
            .unwrap_or(Duration::from_secs(0));

        let progress_ratio = if self.total_tasks > 0 {
            self.completed_tasks as f64 / self.total_tasks as f64
        } else {
            0.0
        };

        let speed = if elapsed.as_secs_f64() > 0.0 {
            self.completed_tasks as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        let eta = if speed > 0.0 && self.completed_tasks < self.total_tasks {
            let remaining = self.total_tasks - self.completed_tasks;
            Duration::from_secs_f64(remaining as f64 / speed)
        } else {
            Duration::from_secs(0)
        };

        ProgressInfo {
            completed: self.completed_tasks,
            total: self.total_tasks,
            progress_ratio,
            elapsed,
            speed,
            eta,
        }
    }

    /// 打印进度信息
    fn print_progress(&self) {
        let info = self.get_progress();
        let percent = info.progress_ratio * 100.0;
        
        println!(
            "进度: {:.1}% ({}/{}) | 速度: {:.0} 任务/秒 | 已用时: {:.1}秒 | 预计剩余: {:.1}秒",
            percent,
            info.completed,
            info.total,
            info.speed,
            info.elapsed.as_secs_f64(),
            info.eta.as_secs_f64()
        );
    }

    /// 获取当前速度
    pub fn get_current_speed(&mut self) -> f64 {
        let now = Instant::now();
        
        if let Some(last_time) = self.last_update_time {
            let time_diff = now.duration_since(last_time).as_secs_f64();
            if time_diff > 0.0 {
                let task_diff = self.completed_tasks - self.last_completed;
                let current_speed = task_diff as f64 / time_diff;
                
                // 更新记录
                self.last_update_time = Some(now);
                self.last_completed = self.completed_tasks;
                
                return current_speed;
            }
        }
        
        0.0
    }

    /// 检查是否完成
    pub fn is_completed(&self) -> bool {
        self.completed_tasks >= self.total_tasks
    }

    /// 获取完成百分比
    pub fn get_percentage(&self) -> f64 {
        if self.total_tasks > 0 {
            (self.completed_tasks as f64 / self.total_tasks as f64) * 100.0
        } else {
            0.0
        }
    }
}

/// 进度信息结构
#[derive(Debug, Clone)]
pub struct ProgressInfo {
    pub completed: usize,
    pub total: usize,
    pub progress_ratio: f64,
    pub elapsed: Duration,
    pub speed: f64,
    pub eta: Duration,
}

impl ProgressInfo {
    /// 格式化为字符串
    pub fn format(&self) -> String {
        format!(
            "进度: {:.1}% ({}/{}) | 速度: {:.0} 任务/秒 | 已用时: {}s | 预计剩余: {}s",
            self.progress_ratio * 100.0,
            self.completed,
            self.total,
            self.speed,
            self.elapsed.as_secs(),
            self.eta.as_secs()
        )
    }

    /// 获取进度条字符串
    pub fn progress_bar(&self, width: usize) -> String {
        let filled = (self.progress_ratio * width as f64) as usize;
        let empty = width - filled;
        
        format!("[{}{}]", "=".repeat(filled), " ".repeat(empty))
    }
}

/// 线程安全的进度跟踪器包装器
pub struct SafeProgressTracker {
    inner: Arc<Mutex<ProgressTracker>>,
}

impl SafeProgressTracker {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(ProgressTracker::new())),
        }
    }

    pub fn start(&self, total_tasks: usize) {
        if let Ok(mut tracker) = self.inner.lock() {
            tracker.start(total_tasks);
        }
    }

    pub fn update(&self, increment: usize) {
        if let Ok(mut tracker) = self.inner.lock() {
            tracker.update(increment);
        }
    }

    pub fn get_progress(&self) -> Option<ProgressInfo> {
        if let Ok(tracker) = self.inner.lock() {
            Some(tracker.get_progress())
        } else {
            None
        }
    }

    pub fn is_completed(&self) -> bool {
        if let Ok(tracker) = self.inner.lock() {
            tracker.is_completed()
        } else {
            false
        }
    }

    pub fn clone_arc(&self) -> Arc<Mutex<ProgressTracker>> {
        Arc::clone(&self.inner)
    }
}

impl Clone for SafeProgressTracker {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// 进度条样式配置
pub struct ProgressBarStyle {
    pub width: usize,
    pub filled_char: char,
    pub empty_char: char,
    pub show_percentage: bool,
    pub show_speed: bool,
    pub show_eta: bool,
}

impl Default for ProgressBarStyle {
    fn default() -> Self {
        Self {
            width: 50,
            filled_char: '=',
            empty_char: ' ',
            show_percentage: true,
            show_speed: true,
            show_eta: true,
        }
    }
}

/// 高级进度跟踪器，支持自定义样式
pub struct AdvancedProgressTracker {
    tracker: ProgressTracker,
    style: ProgressBarStyle,
    last_print_time: Option<Instant>,
    print_interval: Duration,
}

impl AdvancedProgressTracker {
    pub fn new(style: ProgressBarStyle) -> Self {
        Self {
            tracker: ProgressTracker::new(),
            style,
            last_print_time: None,
            print_interval: Duration::from_millis(100), // 100ms更新间隔
        }
    }

    pub fn start(&mut self, total_tasks: usize) {
        self.tracker.start(total_tasks);
        self.last_print_time = Some(Instant::now());
    }

    pub fn update(&mut self, increment: usize) {
        self.tracker.update(increment);
        
        let now = Instant::now();
        let should_print = self.last_print_time
            .map(|last| now.duration_since(last) >= self.print_interval)
            .unwrap_or(true);

        if should_print || self.tracker.is_completed() {
            self.print_advanced_progress();
            self.last_print_time = Some(now);
        }
    }

    fn print_advanced_progress(&self) {
        let info = self.tracker.get_progress();
        let mut output = String::new();

        // 进度条
        let filled = (info.progress_ratio * self.style.width as f64) as usize;
        let empty = self.style.width - filled;
        let bar = format!(
            "[{}{}]",
            self.style.filled_char.to_string().repeat(filled),
            self.style.empty_char.to_string().repeat(empty)
        );
        output.push_str(&bar);

        // 百分比
        if self.style.show_percentage {
            output.push_str(&format!(" {:.1}%", info.progress_ratio * 100.0));
        }

        // 计数
        output.push_str(&format!(" ({}/{})", info.completed, info.total));

        // 速度
        if self.style.show_speed {
            output.push_str(&format!(" | {:.0} 任务/秒", info.speed));
        }

        // 预计剩余时间
        if self.style.show_eta && info.eta.as_secs() > 0 {
            output.push_str(&format!(" | ETA: {}s", info.eta.as_secs()));
        }

        // 使用\r实现原地更新
        print!("\r{}", output);
        std::io::Write::flush(&mut std::io::stdout()).unwrap_or(());

        // 如果完成了，换行
        if self.tracker.is_completed() {
            println!();
        }
    }

    pub fn get_progress(&self) -> ProgressInfo {
        self.tracker.get_progress()
    }

    pub fn is_completed(&self) -> bool {
        self.tracker.is_completed()
    }
}