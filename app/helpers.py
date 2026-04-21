"""Helper functions for pneumonia detection application."""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
from tensorflow.keras.models import Model
from scipy import ndimage


@st.cache_resource(show_spinner=False)
def load_pneumonia_model(model_path: Path) -> Optional[Model]:
    """Load and cache the pneumonia detection model."""
    from tensorflow.keras.models import load_model
    try:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return load_model(str(model_path))
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


@st.cache_data(show_spinner=False)
def load_demo_image(image_path: Path) -> Optional[Image.Image]:
    """Load a demo/sample image from disk with caching."""
    try:
        if not Path(image_path).exists():
            return None
        return Image.open(str(image_path))
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return None


def preprocess_image(img: Image.Image, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Preprocess image for model inference."""
    if not isinstance(img, Image.Image):
        raise TypeError(f"Expected PIL Image, got {type(img)}")
    img = img.resize(target_size)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_arr = np.asarray(img, dtype=np.float32) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr


def create_saliency_map(model: Model, img_array: np.ndarray) -> np.ndarray:
    """Generate a saliency map showing regions of interest."""
    import tensorflow as tf
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
    gradients = tape.gradient(predictions, img_tensor)
    saliency = tf.reduce_max(tf.abs(gradients), axis=-1)[0].numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency


def create_3d_heatmap(
    img_array: np.ndarray,
    saliency_map: np.ndarray,
    is_pneumonia: bool,
    height_scale: float = 35.0,
    downsample_factor: int = 2
) -> go.Figure:
    """Create an interactive 3D heatmap visualization."""
    if img_array.shape[-1] == 3:
        gray_img = np.mean(img_array[0], axis=2)
    else:
        gray_img = img_array[0].squeeze()

    base_surface = (gray_img - gray_img.min()) / (gray_img.max() - gray_img.min() + 1e-8)
    attention_surface = saliency_map
    combined_surface = base_surface * 0.3 + attention_surface * 0.7
    noise = np.random.normal(0, 0.02, combined_surface.shape)
    combined_surface = np.clip(combined_surface + noise, 0, 1)

    downsampled_surface = combined_surface[::downsample_factor, ::downsample_factor]
    downsampled_attention = saliency_map[::downsample_factor, ::downsample_factor]

    h, w = downsampled_surface.shape
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    Z = downsampled_surface * height_scale

    if is_pneumonia:
        colorscale = [
            [0.0, 'rgba(30,30,50,0.8)'],
            [0.2, 'rgba(100,50,50,0.9)'],
            [0.4, 'rgba(180,60,60,0.95)'],
            [0.6, 'rgba(220,80,80,1.0)'],
            [0.8, 'rgba(255,100,100,1.0)'],
            [1.0, 'rgba(255,150,150,1.0)']
        ]
        title = '3D Pneumonia Analysis - Infection Zones Detected'
    else:
        colorscale = [
            [0.0, 'rgba(20,40,30,0.8)'],
            [0.2, 'rgba(40,80,60,0.9)'],
            [0.4, 'rgba(60,120,90,0.95)'],
            [0.6, 'rgba(80,160,120,1.0)'],
            [0.8, 'rgba(100,200,150,1.0)'],
            [1.0, 'rgba(120,220,180,1.0)']
        ]
        title = '3D Lung Analysis - Healthy Tissue Structure'

    surface = go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=downsampled_attention,
        colorscale=colorscale,
        cmin=0, cmax=1,
        showscale=True,
        colorbar=dict(
            title=dict(text='<b>AI Attention & Risk Level</b>', font=dict(size=12, color='#6D4CFF')),
            thickness=20, len=0.7, x=1.08,
            tickfont=dict(size=10, color='#6D4CFF')
        ),
        lighting=dict(ambient=0.4, diffuse=0.8, fresnel=0.3, specular=0.5, roughness=0.2),
        lightposition=dict(x=100, y=200, z=50)
    )

    fig = go.Figure(data=[surface])

    if is_pneumonia:
        threshold = 0.6
        hot_y, hot_x = np.where(downsampled_attention > threshold)
        if len(hot_y) > 0:
            indices = np.random.choice(len(hot_y), min(8, len(hot_y)), replace=False)
            for idx in indices:
                hx, hy = hot_x[idx], hot_y[idx]
                hz = Z[hy, hx] + 2
                fig.add_trace(go.Scatter3d(
                    x=[hx], y=[hy], z=[hz],
                    mode='markers',
                    marker=dict(size=8, color='rgba(255,80,80,0.9)', symbol='diamond'),
                    name='Infection Hotspot'
                ))

    fig.update_layout(
        title=dict(
            text=f'<b>{title}</b>',
            x=0.5, xanchor='center',
            font=dict(size=18, color='#6D4CFF'),
            y=0.98
        ),
        scene=dict(
            xaxis=dict(title='<b>Width</b>', showgrid=True, gridcolor='rgba(109,76,255,0.2)', tickfont=dict(color='#6D4CFF', size=10)),
            yaxis=dict(title='<b>Height</b>', showgrid=True, gridcolor='rgba(109,76,255,0.2)', tickfont=dict(color='#6D4CFF', size=10)),
            zaxis=dict(title='<b>Depth</b>', showgrid=True, gridcolor='rgba(109,76,255,0.2)', tickfont=dict(color='#6D4CFF', size=10)),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.6)),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.6),
            bgcolor='rgba(247,248,255,0.1)'
        ),
        margin=dict(l=0, r=120, t=80, b=0),
        width=950, height=700,
        paper_bgcolor='rgba(247,248,255,0.95)',
        plot_bgcolor='rgba(247,248,255,0.3)',
        font=dict(family='Arial, sans-serif', size=12, color='#6D4CFF'),
        showlegend=False
    )

    return fig


def overlay_saliency_on_image(img: Image.Image, saliency_map: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Overlay saliency map on original image."""
    img_arr = np.array(img.convert("RGB"))
    saliency_normalized = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
    heatmap = np.zeros((*saliency_normalized.shape, 3))
    heatmap[..., 0] = saliency_normalized
    heatmap[..., 1] = 1 - saliency_normalized
    heatmap = (heatmap * 255).astype(np.uint8)
    overlay_img = Image.fromarray(heatmap)
    overlay_img = overlay_img.resize(img.size)
    blended = Image.blend(img.convert("RGB"), overlay_img, alpha)
    return blended


def analyze_affected_regions(saliency_map: np.ndarray, threshold: float = 0.5) -> Optional[Dict[str, Any]]:
    """Analyze and identify affected regions from the saliency map."""
    binary_map = saliency_map > threshold
    labeled_array, num_features = ndimage.label(binary_map)
    
    if num_features > 0:
        sizes = ndimage.sum(binary_map, labeled_array, range(num_features + 1))
        total_affected = np.sum(binary_map)
        max_intensity = np.max(saliency_map)
        mean_intensity = np.mean(saliency_map[binary_map])
        h, w = saliency_map.shape
        affected_coords = np.where(binary_map)
        
        if len(affected_coords[0]) > 0:
            center_h = np.mean(affected_coords[0])
            center_w = np.mean(affected_coords[1])
            location = ""
            if center_h < h/3:
                location += "Upper "
            elif center_h > 2*h/3:
                location += "Lower "
            else:
                location += "Middle "
            if center_w < w/2:
                location += "Left Lung"
            else:
                location += "Right Lung"
        else:
            location = "Diffuse"
        
        return {
            'total_affected_percent': float((total_affected / binary_map.size) * 100),
            'max_intensity': float(max_intensity),
            'mean_intensity': float(mean_intensity),
            'num_regions': int(num_features),
            'primary_location': location
        }
    
    return None


def create_attention_overlay(img: Image.Image, saliency_map: np.ndarray, alpha: float = 0.6) -> Image.Image:
    """Create an attention overlay visualization on the X-ray image."""
    return overlay_saliency_on_image(img, saliency_map, alpha)


def generate_pdf_report(patient_id: str, prediction: str, confidence: float, analysis: Dict[str, Any], saliency_map: np.ndarray, original_image: Image.Image) -> bytes:
    """Generate a comprehensive PDF report for the analysis."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor('#6D4CFF'), spaceAfter=30, alignment=1)
        story.append(Paragraph("Pneumonia Detection Analysis Report", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph(f"<b>Patient ID:</b> {patient_id}", styles['Normal']))
        story.append(Paragraph(f"<b>Diagnosis:</b> {prediction}", styles['Normal']))
        story.append(Paragraph(f"<b>Confidence:</b> {confidence:.1%}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        if analysis:
            story.append(Paragraph("<b>Detailed Analysis:</b>", styles['Heading2']))
            story.append(Paragraph(f"Affected Region: {analysis.get('total_affected_percent', 0):.1f}%", styles['Normal']))
            story.append(Paragraph(f"Primary Location: {analysis.get('primary_location', 'N/A')}", styles['Normal']))
            story.append(Paragraph(f"Number of Regions: {analysis.get('num_regions', 0)}", styles['Normal']))
            story.append(Spacer(1, 0.3*inch))
        
        doc.build(story)
        return pdf_buffer.getvalue()
    except ImportError:
        return b"PDF generation requires reportlab package"
