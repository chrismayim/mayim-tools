<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="4.0.3-Norrköping" styleCategories="AllStyleCategories" hasScaleBasedVisibilityFlag="0" minScale="1e+08" maxScale="0">
  <pipe>
    <provider>
      <resampling enabled="false" zoomedOutResamplingMethod="nearestNeighbour" maxOversampling="2" zoomedInResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer alphaBand="-1" band="1" nodataColor="" opacity="0.7" type="paletted">
      <colorPalette>
        <paletteEntry alpha="255" color="#1a9850" label="Vleis and pans (&lt; 3 %)" value="1"/>
        <paletteEntry alpha="255" color="#d9ef8b" label="Flat (3 to 10 %)" value="2"/>
        <paletteEntry alpha="255" color="#fdae61" label="Hilly (10 to 30 %)" value="3"/>
        <paletteEntry alpha="255" color="#d73027" label="Steep (&gt; 30 %)" value="4"/>
      </colorPalette>
    </rasterrenderer>
    <brightnesscontrast contrast="0" gamma="1" brightness="0"/>
    <huesaturation saturation="0" colorizeOn="0" colorizeStrength="100" grayscaleMode="0" colorizeBlue="128" colorizeRed="255" colorizeGreen="128" invertColors="0"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
