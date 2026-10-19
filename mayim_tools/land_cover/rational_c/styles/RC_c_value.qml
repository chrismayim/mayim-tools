<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="4.0.3-Norrköping" styleCategories="AllStyleCategories" hasScaleBasedVisibilityFlag="0" minScale="1e+08" maxScale="0">
  <pipe>
    <provider>
      <resampling enabled="false" zoomedOutResamplingMethod="nearestNeighbour" maxOversampling="2" zoomedInResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer classificationMin="0" alphaBand="-1" opacity="0.7" nodataColor="" classificationMax="1" band="1" type="singlebandpseudocolor">
      <rastershader>
        <colorrampshader colorRampType="DISCRETE" clip="0" minimumValue="0" labelPrecision="2" maximumValue="1" classificationMode="1">
          <item alpha="255" label="below 0.10" value="0.1" color="#1a9850"/>
          <item alpha="255" label="0.10 to 0.20" value="0.2" color="#91cf60"/>
          <item alpha="255" label="0.20 to 0.30" value="0.3" color="#d9ef8b"/>
          <item alpha="255" label="0.30 to 0.40" value="0.4" color="#fee08b"/>
          <item alpha="255" label="0.40 to 0.50" value="0.5" color="#fdae61"/>
          <item alpha="255" label="0.50 to 0.60" value="0.6" color="#f46d43"/>
          <item alpha="255" label="0.60 to 0.80" value="0.8" color="#d73027"/>
          <item alpha="255" label="0.80 and above" value="inf" color="#7f0000"/>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast contrast="0" gamma="1" brightness="0"/>
    <huesaturation saturation="0" colorizeOn="0" colorizeStrength="100" grayscaleMode="0" colorizeBlue="128" colorizeRed="255" colorizeGreen="128" invertColors="0"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
