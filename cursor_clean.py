# ==================== ADVANCED CURSOR INTERACTION EFFECTS ====================

# Add advanced cursor interaction JavaScript for 3D visualizations
cursor_interaction_js = '''
<script>
document.addEventListener("DOMContentLoaded", function() {
    // Wait for Plotly plots to load
    setTimeout(function() {
        const plots = document.querySelectorAll(".js-plotly-plot");
        
        plots.forEach(plot => {
            let mouseX = 0, mouseY = 0;
            let isHovering = false;
            
            // Mouse move handler for dynamic lighting
            plot.addEventListener("mousemove", function(e) {
                if (!isHovering) return;
                
                const rect = plot.getBoundingClientRect();
                mouseX = (e.clientX - rect.left) / rect.width;
                mouseY = (e.clientY - rect.top) / rect.height;
                
                // Update lighting position based on cursor
                try {
                    const plotDiv = plot.querySelector(".plotly-graph-div");
                    if (plotDiv && plotDiv._fullLayout && plotDiv._fullLayout.scene) {
                        const scene = plotDiv._fullLayout.scene;
                        if (scene.lightposition) {
                            // Dynamic lighting that follows cursor
                            scene.lightposition.x = 100 * (mouseX - 0.5) * 2;
                            scene.lightposition.y = 100 * (mouseY - 0.5) * 2;
                            scene.lightposition.z = 80 + 20 * Math.sin(Date.now() * 0.001);
                            
                            Plotly.relayout(plot, {
                                "scene.lightposition": scene.lightposition
                            });
                        }
                    }
                } catch (error) {
                    // Silently handle any Plotly update errors
                }
            });
            
            // Mouse enter handler - enhance particles and effects
            plot.addEventListener("mouseenter", function() {
                isHovering = true;
                
                // Scale up particles and add glow effect
                const particles = plot.querySelectorAll(".scatterlayer .point");
                particles.forEach(particle => {
                    particle.style.transition = "all 0.3s ease";
                    particle.style.transform = "scale(1.3)";
                    particle.style.filter = "drop-shadow(0 0 8px rgba(109,76,255,0.6))";
                });
                
                // Add subtle pulse animation to surfaces
                const surfaces = plot.querySelectorAll(".surface");
                surfaces.forEach(surface => {
                    surface.style.animation = "pulse 2s infinite";
                });
            });
            
            // Mouse leave handler - reset effects
            plot.addEventListener("mouseleave", function() {
                isHovering = false;
                
                // Reset particle scaling
                const particles = plot.querySelectorAll(".scatterlayer .point");
                particles.forEach(particle => {
                    particle.style.transform = "scale(1)";
                    particle.style.filter = "none";
                });
                
                // Remove pulse animation
                const surfaces = plot.querySelectorAll(".surface");
                surfaces.forEach(surface => {
                    surface.style.animation = "none";
                });
            });
            
            // Add click interaction for camera movement
            plot.addEventListener("click", function(e) {
                const rect = plot.getBoundingClientRect();
                const clickX = (e.clientX - rect.left) / rect.width;
                const clickY = (e.clientY - rect.top) / rect.height;
                
                try {
                    const plotDiv = plot.querySelector(".plotly-graph-div");
                    if (plotDiv && plotDiv._fullLayout && plotDiv._fullLayout.scene) {
                        // Smooth camera transition to clicked point
                        const newCamera = {
                            eye: {
                                x: 1.8 * (clickX - 0.5) * 2,
                                y: 1.8 * (clickY - 0.5) * 2,
                                z: 1.6
                            },
                            center: {x: 0, y: 0, z: 0},
                            up: {x: 0, y: 0, z: 1}
                        };
                        
                        Plotly.animate(plot, {
                            layout: {scene: {camera: newCamera}}
                        }, {
                            duration: 1000,
                            easing: "cubic-in-out"
                        });
                    }
                } catch (error) {
                    // Silently handle animation errors
                }
            });
        });
        
        // Add CSS animations
        const style = document.createElement("style");
        style.textContent = 
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.8; }
            }
            
            .js-plotly-plot {
                transition: all 0.3s ease;
            }
            
            .js-plotly-plot:hover {
                transform: scale(1.02);
                box-shadow: 0 8px 25px rgba(109,76,255,0.15);
            }
        ;
        document.head.appendChild(style);
        
    }, 2000); // Wait 2 seconds for plots to fully load
});
</script>
'''

# Inject the cursor interaction JavaScript
st.components.v1.html(cursor_interaction_js, height=0)
