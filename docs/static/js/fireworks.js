class Firework {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.particles = [];
        this.hue = Math.random() * 360;
        this.sound = new Audio('explosion.mp3'); // Add your sound file
    }

    createParticles(x, y) {
        for (let i = 0; i < 50; i++) {
            this.particles.push({
                x: x,
                y: y,
                vx: Math.random() * 6 - 3,
                vy: Math.random() * 6 - 3,
                radius: Math.random() * 2 + 1,
                alpha: 1,
            });
        }
    }

    draw() {
        this.ctx.globalCompositeOperation = 'destination-out';
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.globalCompositeOperation = 'lighter';

        for (let i = 0; i < this.particles.length; i++) {
            let p = this.particles[i];
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2, false);
            this.ctx.fillStyle = `hsla(${this.hue}, 100%, 50%, ${p.alpha})`;
            this.ctx.fill();

            p.x += p.vx;
            p.y += p.vy;
            p.alpha -= 0.01;

            if (p.alpha <= 0) {
                this.particles.splice(i, 1);
                i--;
            }
        }
    }

    explode(x, y) {
        this.createParticles(x, y);
        this.sound.play();
    }
}

class FireworksDisplay {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.fireworks = [];
        this.isRunning = false;
    }

    start() {
        this.isRunning = true;
        this.animate();
        this.createFireworks();
    }

    stop() {
        this.isRunning = false;
    }

    createFireworks() {
        if (!this.isRunning) return;

        const firework = new Firework(this.canvas);
        const x = Math.random() * this.canvas.width;
        const y = Math.random() * this.canvas.height;
        firework.explode(x, y);
        this.fireworks.push(firework);

        setTimeout(() => this.createFireworks(), Math.random() * 1000 + 500);
    }

    animate() {
        if (!this.isRunning) return;

        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        for (let i = 0; i < this.fireworks.length; i++) {
            this.fireworks[i].draw();
            if (this.fireworks[i].particles.length === 0) {
                this.fireworks.splice(i, 1);
                i--;
            }
        }

        requestAnimationFrame(() => this.animate());
    }
}