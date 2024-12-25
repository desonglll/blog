<template>
  <a :href="href" class="mention">
    <span class="at-symbol">@{{ from }}</span>
    <img :src="icon" class="favicon" v-if="icon" />
    &nbsp;
    {{ text }}
  </a>
</template>

<script setup>
defineProps({
  href: {
    type: String,
    required: true
  },
  text: {
    type: String,
    required: true
  },
  from: {
    type: String,
    required: false
  },
  icon: {
    type: String,
    required: false,
    default: ''
  }
})
</script>

<style scoped>
.mention {
  display: inline-flex;
  align-items: center;
  padding: 0 4px;
  border-radius: 4px;
  background: rgba(35, 131, 226, 0.07);
  color: var(--vp-c-brand);
  text-decoration: none;
  transition: background 0.2s;
}

.mention:hover {
  background: rgba(35, 131, 226, 0.14);
}

.at-symbol {
  margin-right: 2px;
  opacity: 0.7;
}

.favicon {
  width: 16px;
  height: 16px;
  margin-right: 4px;
}
</style>
